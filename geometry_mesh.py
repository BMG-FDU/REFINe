import shutil
from abaqus import *
from abaqusConstants import *
import os, csv, json, logging, mesh
from viewerModules import *
from driverUtils import executeOnCaeStartup
import zlib
import gc
MAP_PATH = r'D:\\CODE\\Abaqus_batch_test\\name_map.json'
SCALE_XY = 10.0
THICKNESS = 1.0

EPS = 1e-9
AREA_TOL = 1e-4

def orient(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def on_segment(a, b, c):
    return (min(a[0],b[0]) - EPS <= c[0] <= max(a[0],b[0]) + EPS and
            min(a[1],b[1]) - EPS <= c[1] <= max(a[1],b[1]) + EPS and
            abs(orient(a,b,c)) <= EPS)

def segments_intersect(p1,p2,p3,p4):
    o1 = orient(p1,p2,p3); o2 = orient(p1,p2,p4)
    o3 = orient(p3,p4,p1); o4 = orient(p3,p4,p2)
    if (o1*o2 < -EPS) and (o3*o4 < -EPS):
        return True
    if abs(o1) <= EPS and on_segment(p1,p2,p3): return True
    if abs(o2) <= EPS and on_segment(p1,p2,p4): return True
    if abs(o3) <= EPS and on_segment(p3,p4,p1): return True
    if abs(o4) <= EPS and on_segment(p3,p4,p2): return True
    return False

def polygon_area(poly):
    a = 0.0
    for i in range(len(poly)-1):
        x1,y1 = poly[i]; x2,y2 = poly[i+1]
        a += x1*y2 - x2*y1
    return 0.5*a

def bbox(poly):
    xs=[p[0] for p in poly]; ys=[p[1] for p in poly]
    return (min(xs), max(xs), min(ys), max(ys))

def bbox_intersect(b1, b2):
    return not (b1[1] < b2[0] or b2[1] < b1[0] or b1[3] < b2[2] or b2[3] < b1[2])

def self_intersect(poly):
    n = len(poly) - 1  # 最后一点与第一点相同
    for i in range(n):
        a1 = poly[i]; a2 = poly[i+1]
        for j in range(i+1, n):
            if j == i+1:
                continue
            if i == 0 and j == n-1:
                continue
            b1 = poly[j]; b2 = poly[j+1]
            if segments_intersect(a1, a2, b1, b2):
                return True
    return False
def polygons_intersect(poly1, poly2):
    if not bbox_intersect(bbox(poly1), bbox(poly2)):
        return False
    for i in range(len(poly1)-1):
        for j in range(len(poly2)-1):
            if segments_intersect(poly1[i], poly1[i+1], poly2[j], poly2[j+1]):
                return True
    return False

def simplify_collinear_closed(poly):
    if len(poly) <= 4:
        return poly
    pts = poly[:-1]
    res = [pts[0]]
    for i in range(1, len(pts)-1):
        a = res[-1]; b = pts[i]; c = pts[i+1]
        ab = (b[0]-a[0], b[1]-a[1]); bc = (c[0]-b[0], c[1]-b[1])
        lab2 = ab[0]*ab[0] + ab[1]*ab[1]
        lbc2 = bc[0]*bc[0] + bc[1]*bc[1]
        if lab2 < EPS or lbc2 < EPS:
            continue
        if abs(orient(a,b,c)) <= EPS:
            continue
        res.append(b)
    res.append(pts[-1])
    res.append(res[0])
    return res

def build_clean_loops(valid):
    loops=[]
    drop_small=0
    drop_self=0
    for cid, pts in valid.items():
        scaled = [(p[0]*SCALE_XY, p[1]*SCALE_XY) for p in pts]
        if scaled[0] != scaled[-1]:
            scaled.append(scaled[0])
        scaled = simplify_collinear_closed(scaled)
        if len(scaled) < 4:
            drop_small += 1
            continue
        a = abs(polygon_area(scaled))
        if a < AREA_TOL:
            drop_small += 1
            continue
        if self_intersect(scaled):
            drop_self += 1
            continue
        loops.append((cid, scaled, a, bbox(scaled)))
    print("Clean step: {} small/degenerate removed, {} self-intersecting removed".format(drop_small, drop_self))
    if not loops:
        return []
    loops.sort(key=lambda x: x[2], reverse=True)
    kept=[]
    for cid, poly, area_v, bb in loops:
        conflict=False
        for _, kpoly, karea, kbb in kept:
            if bbox_intersect(bb, kbb):
                if polygons_intersect(poly, kpoly):
                    conflict=True
                    break
        if not conflict:
            kept.append((cid, poly, area_v, bb))
    print("Clean step: kept {} non-intersecting loops".format(len(kept)))
    return [k[1] for k in kept]

def get_short_id(long_name):
    if os.path.exists(MAP_PATH):
        mapping = json.load(open(MAP_PATH))
    else:
        mapping = {}
    if long_name in mapping:
        return mapping[long_name]
    sid = format(zlib.crc32(long_name.encode()) & 0xffffffff, '08X')
    while sid in mapping.values():
        sid = sid[:-1] + chr((ord(sid[-1]) + 1 - 48) % 10 + 48)
    mapping[long_name] = sid
    json.dump(mapping, open(MAP_PATH, 'w'), indent=4)
    return sid

print("Starting geometry_mesh.py")

def sanitize(n):
    return ''.join(c if c.isalnum() or c == '_' else '_' for c in n)

def setup_logger(dir_):
    lg = logging.getLogger(__name__)
    for h in list(lg.handlers):
        lg.removeHandler(h)
    os.makedirs(dir_, exist_ok=True)
    fh = logging.FileHandler(os.path.join(dir_, 'geometry_mesh.log'))
    ch = logging.StreamHandler()
    fm = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fm); ch.setFormatter(fm)
    lg.addHandler(fh); lg.addHandler(ch); lg.setLevel(logging.DEBUG)
    return lg

def read_contours(csvf):
    print("Reading contours from: {}".format(csvf))
    d = {}
    try:
        with open(csvf) as f:
            r = csv.reader(f); next(r)
            for row in r:
                cid = int(row[0]); x=float(row[1]); y=float(row[2])
                d.setdefault(cid, []).append((x, y))
        xs=[p[0] for v in d.values() for p in v]
        ys=[p[1] for v in d.values() for p in v]
        print("Successfully read {} contours".format(len(d)))
        return d,(min(xs),max(xs),min(ys),max(ys))
    except Exception as e:
        print("Failed to read contours: {}".format(str(e)))
        return {}, (0,0,0,0)

def create_geometry_and_mesh(csvf, out_dir, lg):
    print("Processing: {}".format(csvf))
    base = os.path.splitext(os.path.basename(csvf))[0]
    safe = sanitize(base)
    sid = get_short_id(safe)
    print("Safe name: {}".format(safe))
    try:
        mdl = mdb.Model(name='M_'+sid)
        print("Model created: M_{}".format(safe))
    except Exception as e:
        print("Failed to create model: {}".format(str(e)))
        return None, None, None
    ctr, bounds = read_contours(csvf)
    if not ctr:
        print("No contours found")
        return None, None, None
    valid={}
    for cid, pts in ctr.items():
        uniq=[]
        for i,p in enumerate(pts):
            if i==0 or abs(p[0]-pts[i-1][0])>1e-6 or abs(p[1]-pts[i-1][1])>1e-6:
                uniq.append(p)
        if len(uniq)>=3:
            if uniq[0]!=uniq[-1]: uniq.append(uniq[0])
            valid[cid]=uniq
    print("Valid contours: {}".format(len(valid)))
    try:
        cleaned_loops = build_clean_loops(valid)
        if not cleaned_loops:
            print("No valid loops after cleaning")
            return None, None, None
        sk=mdl.ConstrainedSketch(name='profile', sheetSize=200.)
        line_cnt = 0
        for poly in cleaned_loops:
            for i in range(len(poly)-1):
                sk.Line(point1=poly[i], point2=poly[i+1]); line_cnt += 1
        try:
            print("Sketch profiles detected: {}".format(len(sk.profiles)))
        except:
            pass
        print("Sketch created successfully with {} loops, {} lines".format(len(cleaned_loops), line_cnt))
    except Exception as e:
        print("Failed to create sketch: {}".format(str(e)))
        return None, None, None
    try:
        part=mdl.Part(name='SolidPart', dimensionality=THREE_D, type=DEFORMABLE_BODY)
        part.BaseSolidExtrude(sketch=sk, depth=THICKNESS)
        #######################################
        try:
            del mdl.sketches[sk.name]
        except:
            pass
        sk = None
        #######################################
        print("Part created successfully")
        print("Debug: Part has {} cells after creation".format(len(part.cells)))
    except Exception as e:
        print("Failed to create part: {}".format(str(e)))
        return None, None, None
    try:
        part.setElementType(regions=(part.cells,),
                            elemTypes=(mesh.ElemType(elemCode=C3D8R, elemLibrary=STANDARD),))
        print("Element type set successfully")
    except Exception as e:
        print("Failed to set element type: {}".format(str(e)))
        return None, None, None
    L_raw=bounds[3]-bounds[2]; ok=False
    L_mm  = L_raw * SCALE_XY
    T_mm  = THICKNESS
    for size in (THICKNESS/3, THICKNESS/2, L_mm/20.0):
        try:
            print("Trying mesh size: {}".format(size))
            if part.elements: part.deleteMesh()
            part.seedPart(size=size, deviationFactor=.1, minSizeFactor=.1)
            part.generateMesh()
            if part.elements:
                ok=True
                print("Mesh successful with {} elements".format(len(part.elements)))
                break
        except Exception as e:
            print("Mesh failed for size {}: {}".format(size, str(e)))
    if not ok:
        print("Trying tetrahedral mesh")
        try:
            if part.elements: part.deleteMesh()
            part.setMeshControls(regions=part.cells, elemShape=TET, technique=FREE)
            part.setElementType(regions=(part.cells,),
                                elemTypes=(mesh.ElemType(elemCode=C3D4, elemLibrary=STANDARD),))
            part.seedPart(size=L_mm/10., deviationFactor=.1, minSizeFactor=.1)
            part.generateMesh()
            ok=bool(part.elements)
            if ok:
                print("Tetrahedral mesh successful with {} elements".format(len(part.elements)))
        except Exception as e:
            print("Tetrahedral mesh also failed: {}".format(str(e)))
    if not ok:
        print("All mesh attempts failed")
        return None, None, None
    print("Debug: Part has {} cells after meshing".format(len(part.cells)))
    try:
        a=mdl.rootAssembly
        a.Instance(name='Inst_'+sid, part=part, dependent=ON)
        print("Assembly created successfully")
    except Exception as e:
        print("Failed to create assembly: {}".format(str(e)))
        return None, None, None
    try:
        cae_path=os.path.join(out_dir, safe+'.cae')
        mdb.saveAs(pathName=cae_path)
        print("CAE file saved: {}".format(cae_path))
    except Exception as e:
        print("Failed to save CAE file: {}".format(str(e)))
        return None, None, None
    try:
        job = mdb.Job(name='Job_' + sid, model=mdl.name)
        cwd = os.getcwd()
        os.chdir(out_dir)
        print("Debug: Before writeInput - Part has {} cells, {} elements".format(len(part.cells), len(part.elements)))
        job.writeInput(consistencyChecking=OFF)
        src = os.path.join(out_dir, 'Job_' + sid + '.inp')
        tgt = os.path.join(out_dir, safe + '_base.inp')
        if os.path.exists(src):
            import shutil
            shutil.copyfile(src, tgt)
            with open(tgt, 'r') as r:
                inp_content = r.read()
            print("Debug: INP file size: {} characters".format(len(inp_content)))
            lines = inp_content.split('\n')
            print("Debug: First 15 lines of INP file:")
            for i, line in enumerate(lines[:15]):
                print("  {}: {}".format(i + 1, line))
            important_keywords = ['*NODE', '*ELEMENT', '*NSET', '*ELSET', '*PART']
            print("Debug: Checking for important keywords:")
            for keyword in important_keywords:
                count = inp_content.upper().count(keyword)
                print("  {}: {} occurrences".format(keyword, count))
            print("Input file created: {}".format(tgt))
            os.chdir(cwd)
            return bounds, tgt, safe
        else:
            print("Job input file not found: {}".format(src))
            os.chdir(cwd)
            return None, None, None
    except Exception as e:
        print("Failed to write input file: {}".format(str(e)))
        try:
            os.chdir(cwd)
        except:
            pass
        return None, None, None

def main():
    print("Main function started")
    RES=r'D:\CODE\Abaqus_batch_test\Abaqus_Results'
    CSV=r'D:\CODE\Abaqus_batch_test\Contour_Output'
    print("Result directory: {}".format(RES))
    print("CSV directory: {}".format(CSV))
    print("CSV directory exists: {}".format(os.path.exists(CSV)))
    os.makedirs(RES, exist_ok=True)
    lg=setup_logger(RES)
    try:
        csvs=[os.path.join(CSV,f) for f in os.listdir(CSV)
              if f.endswith('.csv') and not f.endswith('_summary.csv')]
        print("Found {} CSV files".format(len(csvs)))
        for csv_file in csvs[:3]:
            print("  - {}".format(csv_file))
    except Exception as e:
        print("Failed to list CSV files: {}".format(str(e)))
        return
    summary=[]
    for i, c in enumerate(csvs):
        print("\nProcessing {}/{}: {}".format(i+1, len(csvs), c))
        bn=os.path.splitext(os.path.basename(c))[0]
        folder=os.path.join(RES,bn)
        os.makedirs(folder,exist_ok=True)
        print("Output folder: {}".format(folder))
        b,inp,safe=create_geometry_and_mesh(c, folder, lg)
        status = 'SUCCESS' if inp else 'FAILED'
        summary.append({'sample':bn,'status':status, 'bounds':b,'safe':safe})
        print("Status: {}".format(status))
        ###################################
        try:
            Mdb()
            gc.collect()
            print("Mdb reset to free memory")
        except Exception as e:
            print("Mdb reset failed: {}".format(str(e)))
        ###################################
    summary_file = os.path.join(RES,'geometry_mesh_summary.json')
    try:
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
        print("\nSummary saved to: {}".format(summary_file))
        print("Total samples: {}, Successful: {}".format(
            len(summary),
            len([s for s in summary if s['status'] == 'SUCCESS'])
        ))
    except Exception as e:
        print("Failed to save summary: {}".format(str(e)))

if __name__ == '__main__':
    executeOnCaeStartup()
    main()
#abaqus cae noGUI=geometry_mesh.py