from driverUtils import executeOnCaeStartup
executeOnCaeStartup()

import os, glob, sys, datetime, traceback, csv, re, time, json, threading
from abaqus import mdb, openMdb
from abaqusConstants import DEFAULT
from odbAccess import openOdb

LOG = open('batch_submit.log', 'a', encoding='utf-8')
def log(msg=''):
    t = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    s = f'[{t}] {msg}\n'
    LOG.write(s); LOG.flush()
    sys.__stdout__.write(s); sys.__stdout__.flush()

class BatchRunner:                              
    def __init__(self, root, cpus=8, timeout_h=12, delete_odb=False):

        self.root   = root
        self.cpus   = cpus
        self.tmo    = timeout_h * 3600
        self.delodb = delete_odb
        self.result = []                                       


    def run(self):
        cae_files = glob.glob(os.path.join(self.root, '**', '*_simulation.cae'),
                              recursive=True)
        log(f'Discover CAE files {len(cae_files)} 个')
        for i, cae in enumerate(sorted(cae_files), 1):
            log(f'[{i}/{len(cae_files)}] {cae}')
            self._process(cae)

        summary = os.path.join(self.root, 'batch_submit_summary.json')
        json.dump(self.result, open(summary, 'w', encoding='utf-8'), indent=2)
        log(f'Execution completed. Summarize and write {summary}')
        LOG.close()


    def _process(self, cae):
        try:
            try: mdb.close()
            except: pass
            openMdb(cae)
        except Exception as e:
            log(f'  !! cannot open CAE: {e}')
            self.result.append({'case': os.path.basename(os.path.dirname(cae)),
                                'cae': cae, 'job': None, 'status': 'CAE_OPEN_ERROR'})
            return

        jobs = [j for j in mdb.jobs.values() if j.name.upper().startswith('JOBSIM_')]
        if not jobs:
            log('  !! Job not found')
            self.result.append({'case': os.path.basename(os.path.dirname(cae)),
                                'cae': cae, 'job': None, 'status': 'NO_JOB'})
            return

        wd = os.path.dirname(cae)
        os.chdir(wd)

        for job in jobs:
            csv_path = os.path.join(wd, job.name + '_RF.csv')
            if os.path.exists(csv_path):
                log(f'The CSV already exists. Skip {job.name}')
                self.result.append({'case': os.path.basename(os.path.dirname(cae)),
                                    'cae' : cae,
                                    'job' : job.name,
                                    'status': 'SKIP'})
                continue

            job_info = {'case' : os.path.basename(os.path.dirname(cae)),
                        'cae'  : cae,
                        'job'  : job.name}

            try:

                job.setValues(numCpus=self.cpus, numDomains=self.cpus,
                              multiprocessingMode=DEFAULT,
                              numThreadsPerMpiProcess=1,
                              getMemoryFromAnalysis=True)
                log(f'  Submit {job.name} (CPU={self.cpus})')
                th = threading.Thread(target=job.waitForCompletion)
                job.submit(consistencyChecking=OFF)
                th.start()
                th.join(self.tmo)

                if th.is_alive():                      
                    try: job.abort()
                    except: pass
                    log('  !! TIMEOUT')
                    job_info['status'] = 'TIMEOUT'
                    self.result.append(job_info); continue

                odb_path = os.path.join(wd, job.name + '.odb')
                last = -1; stable = 0
                while True:
                    cur = os.path.getsize(odb_path) if os.path.exists(odb_path) else -1
                    if cur == last and not os.path.exists(odb_path + '.lck') and cur > 0:
                        stable += 1
                        if stable >= 3: break
                    else:
                        stable = 0
                    last = cur
                    time.sleep(5)

                if not os.path.exists(odb_path):
                    log('  !! NO ODB ')
                    job_info['status'] = 'NO_ODB'
                    self.result.append(job_info); continue

                if self._extract_curve(odb_path, os.path.basename(csv_path)):
                    if self.delodb:
                        try: os.remove(odb_path)
                        except: pass
                    job_info['status'] = 'OK'
                else:
                    job_info['status'] = 'EXTRACT_FAIL'

            except Exception as e:
                log(f'  !! {e}')
                traceback.print_exc(file=LOG)
                job_info['status'] = 'ERROR'

            self.result.append(job_info)

    def _extract_curve(self, odb_path, csv_name):
        try:
            odb = openOdb(odb_path)

            step = None
            if 'LoadStep' in odb.steps:
                step = odb.steps['LoadStep']
            else:
                steps = list(odb.steps.values())
                if len(steps) > 1 and steps[-1].name.lower() != 'initial':
                    step = steps[-1]
                else:
                    step = steps[0]

            rf = u = None
            for reg in step.historyRegions.values():
                n = reg.name.upper()
                if rf is None and 'RF2' in reg.historyOutputs and re.search(r'BOTTOM|REFBOT', n):
                    rf = reg.historyOutputs['RF2'].data
                if u is None and 'U2' in reg.historyOutputs and re.search(r'TOP|REFTOP', n):
                    u = reg.historyOutputs['U2'].data
                if rf and u: break                          

            if rf is None:
                for reg in step.historyRegions.values():
                    if 'RF2' in reg.historyOutputs:
                        rf = reg.historyOutputs['RF2'].data; break
            if u is None:
                for reg in step.historyRegions.values():
                    if 'U2' in reg.historyOutputs:
                        u = reg.historyOutputs['U2'].data; break

            if rf is None or u is None:
                odb.close(); return False

            n = min(len(rf), len(u))
            csv_path = os.path.join(os.path.dirname(odb_path), csv_name)
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Time', 'Disp', 'Force'])
                for (t, fce), (_, disp) in zip(rf[:n], u[:n]):
                    writer.writerow([t, disp, fce])

            odb.close(); return True

        except Exception as e:
            log(f'  !! 曲线提取错误: {e}')
            traceback.print_exc(file=LOG)
            try: odb.close()
            except: pass
            return False


if __name__ == '__main__':
    ROOT = r'D:\CODE\Abaqus_batch_test\Abaqus_Results' 
    BatchRunner(ROOT, cpus=8, timeout_h=12, delete_odb=False).run()
# abaqus cae noGUI=batch_submit.py