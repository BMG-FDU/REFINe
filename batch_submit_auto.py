from driverUtils import executeOnCaeStartup
executeOnCaeStartup()

import os, glob, sys, datetime, traceback, csv, re, time, json, threading
from abaqus import mdb, openMdb
from abaqusConstants import DEFAULT, OFF
from odbAccess import openOdb

LOG = None

def init_log():
    global LOG
    LOG = open('batch_submit_auto.log', 'a')

def safe_str(s):
    try:
        return str(s)
    except:
        try:
            if hasattr(s, 'encode'):
                return s.encode('utf-8')
        except:
            pass
        return repr(s)

def log(msg=''):
    if LOG is None:
        init_log()
    t = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    s = '[{0}] {1}\n'.format(t, safe_str(msg))
    try:
        LOG.write(s)
        LOG.flush()
    except:
        pass
    try:
        sys.__stdout__.write(s)
        sys.__stdout__.flush()
    except:
        pass

class BatchRunner:
    def __init__(self, root, cpus=8, timeout_h=12, delete_odb=False, resume_mode='missing_shear'):
        self.root   = root
        self.cpus   = cpus
        self.tmo    = timeout_h * 3600
        self.delodb = delete_odb
        self.resume_mode = resume_mode
        self.result = []

    def run(self):
        log('Searching CAE files, root: {0}'.format(self.root))
        log('Resume mode: {0}'.format(self.resume_mode))

        cae_files = []
        for root, dirs, files in os.walk(self.root):
            for f in files:
                if f.endswith('.cae'):
                    full_path = os.path.join(root, f)
                    cae_files.append(full_path)

        log('Found {0} CAE files'.format(len(cae_files)))

        sim_files = [c for c in cae_files if 'simulation' in os.path.basename(c).lower()]
        log('Files containing "simulation": {0}'.format(len(sim_files)))

        classified = {'tensile': [], 'compression': [], 'shear': [], 'contact': []}

        for cae in sim_files:
            sim_type = self._get_sim_type(cae)
            if sim_type in classified:
                classified[sim_type].append(cae)

        log('')
        log('Classification summary:')
        for stype, files in classified.items():
            log('  - {0}: {1}'.format(stype, len(files)))

        if self.resume_mode:
            classified = self._filter_by_resume_mode(classified)
            log('')
            log('After resume mode filter:')
            for stype, files in classified.items():
                log('  - {0}: {1}'.format(stype, len(files)))

        all_ordered = (sorted(classified['tensile']) +
                      sorted(classified['compression']) +
                      sorted(classified['shear']) +
                      sorted(classified['contact']))

        if not all_ordered:
            log('')
            log('No tasks to run')
            if LOG:
                LOG.close()
            return

        log('')
        log('Processing {0} files'.format(len(all_ordered)))
        log('')

        for i, cae in enumerate(all_ordered, 1):
            sim_type = self._get_sim_type(cae)
            log('[{0}/{1}] [{2}] {3}'.format(i, len(all_ordered), sim_type.upper(), os.path.basename(cae)))
            self._process(cae, sim_type)

        summary = os.path.join(self.root, 'batch_submit_auto_summary.json')
        with open(summary, 'w') as f:
            json.dump(self.result, f, indent=2, ensure_ascii=False)

        stats = {}
        for stype in ['tensile', 'compression', 'shear', 'contact']:
            stats[stype] = {'ok': 0, 'fail': 0}

        for r in self.result:
            stype = r.get('sim_type', 'unknown')
            if stype in stats:
                if r['status'] == 'OK':
                    stats[stype]['ok'] += 1
                else:
                    stats[stype]['fail'] += 1

        log('')
        log('='*60)
        log('Final summary:')
        for stype in ['tensile', 'compression', 'shear', 'contact']:
            log('  {0}: OK={1}, FAIL={2}'.format(stype, stats[stype]['ok'], stats[stype]['fail']))
        log('Summary file: {0}'.format(summary))
        if LOG:
            LOG.close()

    def _filter_by_resume_mode(self, classified):
        if self.resume_mode == 'missing_shear':
            return self._filter_missing_shear(classified)
        elif self.resume_mode == 'missing_contact':
            return self._filter_missing_contact(classified)
        elif self.resume_mode == 'missing_any':
            return self._filter_missing_any(classified)
        else:
            return classified

    def _filter_missing_shear(self, classified):
        log('')
        log('Checking for missing shear simulations...')

        filtered_shear = []
        total = len(classified['shear'])

        for idx, shear_cae in enumerate(classified['shear'], 1):
            try:
                case_dir = os.path.dirname(shear_cae)
                case_name = os.path.basename(case_dir)

                log('  [{0}/{1}] Checking: {2}'.format(idx, total, case_name))

                has_compression = False
                try:
                    compression_csv_pattern = os.path.join(case_dir, 'JobSim_*_compression_RF.csv')
                    compression_csvs = glob.glob(compression_csv_pattern)
                    if compression_csvs:
                        has_compression = True
                except Exception as e:
                    log('    Warning: failed to search compression CSV: {0}'.format(str(e)))

                has_shear = False
                try:
                    shear_csv_pattern = os.path.join(case_dir, 'JobSim_*_shear_RF.csv')
                    shear_csvs = glob.glob(shear_csv_pattern)
                    if shear_csvs:
                        has_shear = True
                except Exception as e:
                    log('    Warning: failed to search shear CSV: {0}'.format(str(e)))

                if has_compression and not has_shear:
                    filtered_shear.append(shear_cae)
                    log('    -> Need to run (has compression, missing shear)')
                elif has_shear:
                    log('    -> Skip (shear CSV exists)')
                elif not has_compression:
                    log('    -> Skip (no compression CSV)')

            except Exception as e:
                log('  Error: {0}'.format(str(e)))
                continue

        log('')
        log('Found {0} shear simulations to run'.format(len(filtered_shear)))

        return {
            'tensile': [],
            'compression': [],
            'shear': filtered_shear,
            'contact': []
        }

    def _filter_missing_contact(self, classified):
        log('')
        log('Checking for missing contact simulations...')

        filtered_contact = []
        total = len(classified['contact'])

        for idx, contact_cae in enumerate(classified['contact'], 1):
            try:
                case_dir = os.path.dirname(contact_cae)
                case_name = os.path.basename(case_dir)

                log('  [{0}/{1}] Checking: {2}'.format(idx, total, case_name))

                has_basic = False
                try:
                    compression_csvs = glob.glob(os.path.join(case_dir, 'JobSim_*_compression_RF.csv'))
                    shear_csvs = glob.glob(os.path.join(case_dir, 'JobSim_*_shear_RF.csv'))
                    if compression_csvs or shear_csvs:
                        has_basic = True
                except Exception as e:
                    log('    Warning: failed to search basic CSV: {0}'.format(str(e)))

                has_contact = False
                try:
                    contact_csvs = glob.glob(os.path.join(case_dir, 'JobSim_*_contact_RF.csv'))
                    if contact_csvs:
                        has_contact = True
                except Exception as e:
                    log('    Warning: failed to search contact CSV: {0}'.format(str(e)))

                if has_basic and not has_contact:
                    filtered_contact.append(contact_cae)
                    log('    -> Need to run (has basic simulation, missing contact)')
                elif has_contact:
                    log('    -> Skip (contact CSV exists)')
                elif not has_basic:
                    log('    -> Skip (no basic simulation CSV)')

            except Exception as e:
                log('  Error: {0}'.format(str(e)))
                continue

        log('')
        log('Found {0} contact simulations to run'.format(len(filtered_contact)))

        return {
            'tensile': [],
            'compression': [],
            'shear': [],
            'contact': filtered_contact
        }

    def _filter_missing_any(self, classified):
        log('')
        log('Checking all missing simulations...')

        filtered = {'tensile': [], 'compression': [], 'shear': [], 'contact': []}

        csv_suffixes = {
            'tensile': '_RF.csv',
            'compression': '_compression_RF.csv',
            'shear': '_shear_RF.csv',
            'contact': '_contact_RF.csv'
        }

        for sim_type in ['tensile', 'compression', 'shear', 'contact']:
            total = len(classified[sim_type])
            for idx, cae in enumerate(classified[sim_type], 1):
                try:
                    case_dir = os.path.dirname(cae)
                    case_name = os.path.basename(case_dir)

                    log('  [{0}/{1}] Checking {2}: {3}'.format(idx, total, sim_type, case_name))

                    csv_pattern = os.path.join(case_dir, 'JobSim_*{0}'.format(csv_suffixes[sim_type]))
                    csvs = glob.glob(csv_pattern)

                    if not csvs:
                        filtered[sim_type].append(cae)
                        log('    -> Need to run (CSV missing)')
                    else:
                        log('    -> Skip (CSV exists)')

                except Exception as e:
                    log('  Error: {0}'.format(str(e)))
                    continue

        return filtered

    def _get_sim_type(self, cae_path):
        basename = os.path.basename(cae_path).lower()

        if 'contact' in basename:
            return 'contact'
        elif 'compress' in basename:
            return 'compression'
        elif 'shear' in basename:
            return 'shear'
        elif 'simulation' in basename:
            return 'tensile'
        else:
            return 'unknown'

    def _process(self, cae, sim_type):
        try:
            try: mdb.close()
            except: pass
            openMdb(cae)
        except Exception as e:
            log('  !! Cannot open CAE: {0}'.format(str(e)))
            self.result.append({'case': os.path.basename(os.path.dirname(cae)),
                                'cae': cae, 'job': None, 'sim_type': sim_type,
                                'status': 'CAE_OPEN_ERROR'})
            return

        jobs = [j for j in mdb.jobs.values() if j.name.upper().startswith('JOBSIM')]
        if not jobs:
            log('  !! No Job found (name should start with JobSim)')
            self.result.append({'case': os.path.basename(os.path.dirname(cae)),
                                'cae': cae, 'job': None, 'sim_type': sim_type,
                                'status': 'NO_JOB'})
            return

        wd = os.path.dirname(cae)
        os.chdir(wd)

        for job in jobs:
            job_info = {'case' : os.path.basename(os.path.dirname(cae)),
                        'cae'  : cae,
                        'job'  : job.name,
                        'sim_type': sim_type}

            try:
                job.setValues(numCpus=self.cpus, numDomains=self.cpus,
                              multiprocessingMode=DEFAULT,
                              numThreadsPerMpiProcess=1,
                              getMemoryFromAnalysis=True)

                log('  Submit {0} ({1}, CPU={2})'.format(job.name, sim_type, self.cpus))

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
                    log('  !! ODB not found')
                    job_info['status'] = 'NO_ODB'
                    self.result.append(job_info); continue

                csv_suffix = {'compression': '_compression_RF.csv',
                             'shear': '_shear_RF.csv',
                             'contact': '_contact_RF.csv',
                             'tensile': '_RF.csv'}
                csv_name = job.name + csv_suffix.get(sim_type, '_RF.csv')

                if self._extract_curve(odb_path, csv_name, sim_type):
                    if self.delodb:
                        try: os.remove(odb_path)
                        except: pass
                    job_info['status'] = 'OK'
                    job_info['csv'] = csv_name
                    log('  OK -> {0}'.format(csv_name))
                else:
                    job_info['status'] = 'EXTRACT_FAIL'
                    log('  !! Data extraction failed')

            except Exception as e:
                log('  !! {0}'.format(str(e)))
                try:
                    traceback.print_exc(file=LOG)
                except:
                    pass
                job_info['status'] = 'ERROR'

            self.result.append(job_info)

    def _extract_curve(self, odb_path, csv_name, sim_type):
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

            if sim_type == 'shear':
                rf_var = 'RF1'
                u_var = 'U1'
            else:
                rf_var = 'RF2'
                u_var = 'U2'

            rf = u = None

            for reg in step.historyRegions.values():
                n = reg.name.upper()
                if rf is None and rf_var in reg.historyOutputs:
                    if re.search(r'BOTTOM|REFBOT|LEFT|REFLEFT', n):
                        rf = reg.historyOutputs[rf_var].data
                if u is None and u_var in reg.historyOutputs:
                    if re.search(r'TOP|REFTOP|RIGHT|REFRIGHT', n):
                        u = reg.historyOutputs[u_var].data
                if rf and u: break

            if rf is None:
                for reg in step.historyRegions.values():
                    if rf_var in reg.historyOutputs:
                        rf = reg.historyOutputs[rf_var].data
                        log('    Found {0} in: {1}'.format(rf_var, reg.name))
                        break

            if u is None:
                for reg in step.historyRegions.values():
                    if u_var in reg.historyOutputs:
                        u = reg.historyOutputs[u_var].data
                        log('    Found {0} in: {1}'.format(u_var, reg.name))
                        break

            if rf is None or u is None:
                log('  !! Missing data: RF={0}, U={1}'.format(rf is not None, u is not None))
                log('  Available history regions:')
                for reg in step.historyRegions.values():
                    outputs = list(reg.historyOutputs.keys())
                    if len(outputs) > 5:
                        log('    {0}: {1}...'.format(reg.name, outputs[:5]))
                    else:
                        log('    {0}: {1}'.format(reg.name, outputs))
                odb.close()
                return False

            n = min(len(rf), len(u))
            csv_path = os.path.join(os.path.dirname(odb_path), csv_name)

            with open(csv_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Time', 'Disp', 'Force'])
                for (t, fce), (_, disp) in zip(rf[:n], u[:n]):
                    writer.writerow([t, disp, fce])

            odb.close()
            return True

        except Exception as e:
            log('  !! Curve extraction error: {0}'.format(str(e)))
            try:
                traceback.print_exc(file=LOG)
            except:
                pass
            try: odb.close()
            except: pass
            return False

if __name__ == '__main__':
    init_log()
    ROOT = r''

    BatchRunner(ROOT, cpus=8, timeout_h=12, delete_odb=False,
                resume_mode='missing_any').run()