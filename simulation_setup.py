from abaqus import *
from abaqusConstants import *
import os, json, logging, sys, time, shutil, glob, zlib, gc
import mesh

MAP_PATH = r'D:\\CODE\\Abaqus_batch_test\\name_map.json'

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

print("Script started - simulation_setup.py")

def sanitize(n):
    return ''.join(c if c.isalnum() or c == '_' else '_' for c in n)

def logger(root):
    lg = logging.getLogger(__name__)
    for h in list(lg.handlers):
        lg.removeHandler(h)
    os.makedirs(root, exist_ok=True)
    fh = logging.FileHandler(os.path.join(root, 'simulation_setup.log'))
    ch = logging.StreamHandler(sys.stdout)
    fm = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fm)
    ch.setFormatter(fm)
    lg.addHandler(fh)
    lg.addHandler(ch)
    lg.setLevel(logging.DEBUG)
    return lg

def setup_boundary_conditions(mdl, instance, x_min, x_max, y_min, y_max, L_total, delta, z_mid):
    a = mdl.rootAssembly
    mid_x = (x_min + x_max) / 2.0
    bottom_10pct_max = y_min + 0.1 * L_total
    top_10pct_min = y_max - 0.1 * L_total
    rp_top = a.ReferencePoint(point=(mid_x, y_max, z_mid))
    rp_bottom = a.ReferencePoint(point=(mid_x, y_min, z_mid))
    ref_top = a.Set(referencePoints=(a.referencePoints[rp_top.id],), name='RefTop')
    ref_bot = a.Set(referencePoints=(a.referencePoints[rp_bottom.id],), name='RefBot')
    use_node_approach = False
    try:
        all_faces = list(instance.faces)
        bottom_faces = []
        top_faces = []
        for face in all_faces:
            vertices = [instance.vertices[v].pointOn[0] for v in face.getVertices()]
            y_coords = [v[1] for v in vertices]
            min_y_face = min(y_coords)
            max_y_face = max(y_coords)
            centroid_y_face = sum(y_coords) / len(y_coords)
            face_height = max_y_face - min_y_face
            if face_height == 0:
                if centroid_y_face <= bottom_10pct_max:
                    bottom_faces.append(face)
                elif centroid_y_face >= top_10pct_min:
                    top_faces.append(face)
                continue
            bottom_portion = max(0, min(bottom_10pct_max - min_y_face, face_height)) / face_height
            top_portion = max(0, min(max_y_face - top_10pct_min, face_height)) / face_height
            if bottom_portion > 0.6:
                bottom_faces.append(face)
            elif top_portion > 0.6:
                top_faces.append(face)
            elif min_y_face <= bottom_10pct_max and max_y_face <= bottom_10pct_max:
                bottom_faces.append(face)
            elif min_y_face >= top_10pct_min and max_y_face >= top_10pct_min:
                top_faces.append(face)
            else:
                if centroid_y_face <= bottom_10pct_max:
                    bottom_faces.append(face)
                elif centroid_y_face >= top_10pct_min:
                    top_faces.append(face)
        print("Faces: bottom={}, top={}".format(len(bottom_faces), len(top_faces)))
        if len(bottom_faces) > 0 and len(top_faces) > 0:
            bottom_face_seq = [instance.faces[f.index:f.index + 1] for f in bottom_faces]
            top_face_seq = [instance.faces[f.index:f.index + 1] for f in top_faces]
            all_cells = list(instance.cells)
            bottom_cells = []
            top_cells = []
            for cell in all_cells:
                cell_vertices = [instance.vertices[v].pointOn[0] for v in cell.getVertices()]
                y_coords_cell = [v[1] for v in cell_vertices]
                min_y_cell = min(y_coords_cell)
                max_y_cell = max(y_coords_cell)
                centroid_y_cell = sum(y_coords_cell) / len(y_coords_cell)
                cell_height = max_y_cell - min_y_cell
                if cell_height == 0:
                    continue
                if min_y_cell <= bottom_10pct_max and max_y_cell <= bottom_10pct_max:
                    bottom_cells.append(cell)
                elif centroid_y_cell <= bottom_10pct_max and (max_y_cell - bottom_10pct_max) / cell_height < 0.5:
                    bottom_cells.append(cell)
                if min_y_cell >= top_10pct_min and max_y_cell >= top_10pct_min:
                    top_cells.append(cell)
                elif centroid_y_cell >= top_10pct_min and (top_10pct_min - min_y_cell) / cell_height < 0.5:
                    top_cells.append(cell)
            if len(bottom_cells) > 0:
                bottom_cell_seq = [instance.cells[c.index:c.index + 1] for c in bottom_cells]
                bottom_cell_set = a.Set(cells=bottom_cell_seq, name='Bottom_Cell_Set')
                mdl.RigidBody(name='Rigid-Bottom', refPointRegion=ref_bot, bodyRegion=bottom_cell_set)
            else:
                bottom_surface = a.Surface(name='Bottom_Surface', side1Faces=bottom_face_seq)
                mdl.Coupling(name='Coupling-Bottom', controlPoint=ref_bot, surface=bottom_surface,
                             influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, localCsys=None,
                             u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)
            if len(top_cells) > 0:
                top_cell_seq = [instance.cells[c.index:c.index + 1] for c in top_cells]
                top_cell_set = a.Set(cells=top_cell_seq, name='Top_Cell_Set')
                mdl.RigidBody(name='Rigid-Top', refPointRegion=ref_top, bodyRegion=top_cell_set)
            else:
                top_surface = a.Surface(name='Top_Surface', side1Faces=top_face_seq)
                mdl.Coupling(name='Coupling-Top', controlPoint=ref_top, surface=top_surface,
                             influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, localCsys=None,
                             u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)
        else:
            use_node_approach = True
    except Exception as e:
        print("Face-based BC setup failed: {}".format(str(e)))
        use_node_approach = True
    if use_node_approach:
        print("Using node-based BC approach.")
        all_nodes_bc = instance.nodes
        bottom_nodes_bc = []
        top_nodes_bc = []
        for node_bc in all_nodes_bc:
            y_coord_bc = node_bc.coordinates[1]
            if y_coord_bc <= bottom_10pct_max:
                bottom_nodes_bc.append(node_bc)
            elif y_coord_bc >= top_10pct_min:
                top_nodes_bc.append(node_bc)
        if len(bottom_nodes_bc) == 0 or len(top_nodes_bc) == 0:
            print("ERROR: Node-based BC approach failed: bottom or top nodes not found.")
            return False
        print("Found {} bottom and {} top nodes for BCs.".format(len(bottom_nodes_bc), len(top_nodes_bc)))
        try:
            bottom_node_set_bc = a.Set(nodes=mesh.MeshNodeArray(nodes=bottom_nodes_bc), name='Bottom_Nodes_BC_Set')
            top_node_set_bc = a.Set(nodes=mesh.MeshNodeArray(nodes=top_nodes_bc), name='Top_Nodes_BC_Set')
            mdl.RigidBody(name='Rigid-Bottom-Nodes', refPointRegion=ref_bot, tieRegion=bottom_node_set_bc)
            mdl.RigidBody(name='Rigid-Top-Nodes', refPointRegion=ref_top, tieRegion=top_node_set_bc)
        except Exception as e_node_bc:
            print("Node-based BC with RPs failed: {}".format(str(e_node_bc)))
            try:
                mdl.EncastreBC(name='BC-Bottom-Direct', createStepName='Initial', region=bottom_node_set_bc)
                mdl.DisplacementBC(name='BC-Top-Direct', createStepName='LoadStep', region=top_node_set_bc,
                                   u1=0.0, u2=delta, u3=0.0, amplitude='RampAmplitude', fixed=OFF)
                mdl.HistoryOutputRequest(name='H-Output-Bottom-RF-Direct', createStepName='LoadStep',
                                         region=bottom_node_set_bc, variables=('RF2', 'RT2'), summation=TOTAL)
                mdl.HistoryOutputRequest(name='H-Output-Top-U-Direct', createStepName='LoadStep',
                                         region=top_node_set_bc, variables=('U2',), summation=NONE)
                return True
            except Exception as e_direct_bc:
                print("Direct BC approach also failed: {}".format(str(e_direct_bc)))
                return False
    mdl.EncastreBC(name='BC-Bottom-Fixed', createStepName='Initial', region=ref_bot)
    mdl.DisplacementBC(name='BC-Top-Displacement', createStepName='LoadStep', region=ref_top,
                       u1=0.0, u2=delta, u3=0.0, ur1=0.0, ur2=0.0, ur3=0.0,
                       amplitude='RampAmplitude', fixed=OFF)
    mdl.HistoryOutputRequest(name='H-Output-Bottom-RF', createStepName='LoadStep',
                             region=ref_bot, variables=('RF2',), frequency=1)
    mdl.HistoryOutputRequest(name='H-Output-Top-RF', createStepName='LoadStep',
                             region=ref_top, variables=('RF2',), frequency=1)
    mdl.HistoryOutputRequest(name='H-Output-Top-U', createStepName='LoadStep',
                             region=ref_top, variables=('U2',), frequency=1)
    return True

def create_simulation(cae_file, base_inp, sim_inp, sample_name):
    print("Creating simulation for: {}".format(sample_name))
    print("CAE file: {}".format(cae_file))
    print("Base input: {}".format(base_inp))
    print("Output input: {}".format(sim_inp))
    safe_name = sanitize(sample_name)
    sid = get_short_id(safe_name)
    model_name = 'SimModel_' + safe_name
    try:
        if os.path.exists(cae_file):
            openMdb(cae_file)
            original_model = None
            for model in mdb.models.values():
                if hasattr(model, 'parts') and len(model.parts) > 0:
                    original_model = model
                    break
            if original_model is None:
                mdl = mdb.ModelFromInputFile(name=model_name, inputFileName=base_inp)
            else:
                mdl = mdb.Model(name=model_name, objectToCopy=original_model)
        else:
            mdl = mdb.ModelFromInputFile(name=model_name, inputFileName=base_inp)
    except Exception as e:
        print("Failed to load model: {}".format(str(e)))
        return False
    if hasattr(mdl, 'rigidBodies'):
        for k in list(mdl.rigidBodies.keys()):
            del mdl.rigidBodies[k]
    for k in list(mdl.boundaryConditions.keys()):
        del mdl.boundaryConditions[k]
    for k in list(mdl.steps.keys()):
        if k != 'Initial':
            del mdl.steps[k]
    for k in list(mdl.amplitudes.keys()):
        del mdl.amplitudes[k]
    part = mdl.parts.values()[0]
    a = mdl.rootAssembly
    instance_name = 'Inst_' + sid
    if len(a.instances) > 0:
        old_name = list(a.instances.keys())[0]
        if old_name != instance_name:
            a.features.changeKey(fromName=old_name, toName=instance_name)
    else:
        a.Instance(name=instance_name, part=part, dependent=ON)
    instance = a.instances[instance_name]
    try:
        if hasattr(part, 'setElementType') and len(part.cells) > 0:
            elemTypes = (mesh.ElemType(elemCode=C3D8H, elemLibrary=STANDARD,
                                       secondOrderAccuracy=OFF, distortionControl=DEFAULT),)
            part.setElementType(regions=(part.cells,), elemTypes=elemTypes)
    except Exception as e:
        print("Warning: Could not set element types: {}".format(str(e)))
    try:
        mat = mdl.Material(name='Mat_TPU95A')
        mat.Density(table=((1160.0,),))
        mat.Hyperelastic(materialType=ISOTROPIC, testData=OFF, type=MOONEY_RIVLIN,
                         volumetricResponse=POISSON_RATIO, table=((6.0, 3.0, 0.495),))
        plastic_data = []#← use really data
        if plastic_data:
            mat.Plastic(table=tuple(plastic_data))
        print("Material created successfully")
    except Exception as e:
        print("Failed to create material: {}".format(str(e)))
        return False
    try:
        if len(part.elements) == 0:
            part.seedPart(size=0.3)
            part.generateMesh()
        elem_set = part.Set(name='Set_AllElems', elements=part.elements)
        if 'Section_TPU95A' not in mdl.sections:
            mdl.HomogeneousSolidSection(name='Section_TPU95A', material='Mat_TPU95A', thickness=None)
        part.SectionAssignment(region=elem_set, sectionName='Section_TPU95A',
                               offset=0.0, offsetType=MIDDLE_SURFACE,
                               offsetField='', thicknessAssignment=FROM_SECTION)
        print("Section assigned to {} elements".format(len(part.elements)))
    except Exception as e:
        print("Failed to assign section: {}".format(str(e)))
        return False
    try:
        xs = [n.coordinates[0] for n in instance.nodes]
        ys = [n.coordinates[1] for n in instance.nodes]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        L_total = y_max - y_min
        delta = 1.2 * L_total
        z_mid = 0.05
    except Exception as e:
        print("Failed to setup assembly: {}".format(str(e)))
        return False
    try:
        load_factor = 1.0
        num_points = 20
        new_timePeriod = load_factor
        inc = new_timePeriod / float(num_points)
        new_initialInc = inc
        new_maxInc = inc
        amp_data = tuple((i * new_timePeriod / float(num_points), i / float(num_points))
                         for i in range(num_points + 1))
        mdl.TabularAmplitude(name='RampAmplitude', timeSpan=STEP, smooth=SOLVER_DEFAULT, data=amp_data)
        mdl.StaticStep(name='LoadStep', previous='Initial', timePeriod=new_timePeriod,
                       initialInc=new_initialInc, minInc=1e-7, maxInc=new_maxInc, maxNumInc=10000,
                       nlgeom=ON, stabilizationMethod=DISSIPATED_ENERGY_FRACTION)
        mdl.fieldOutputRequests['F-Output-1'].setValues(variables=('S', 'U', 'RF', 'COORD'), numIntervals=num_points)
        mdl.HistoryOutputRequest(name='H-Output-Force', createStepName='LoadStep',
                                 variables=('ALLSE', 'ALLSD'), region=MODEL)
    except Exception as e:
        print("Failed to create step: {}".format(str(e)))
        return False
    bc_success = setup_boundary_conditions(mdl, instance, x_min, x_max, y_min, y_max, L_total, delta, z_mid)
    if not bc_success:
        print("Failed to setup boundary conditions")
        return False
    try:
        timestamp = str(int(time.time() * 1000))[-6:]
        tmp_name = 'TMP_' + sid + '_' + timestamp
        out_dir = os.path.dirname(sim_inp)
        orig_dir = os.getcwd()
        os.chdir(out_dir)
        tmp_job = mdb.Job(name=tmp_name, model=mdl.name)
        tmp_job.writeInput(consistencyChecking=OFF)
        tmp_inp = tmp_name + '.inp'
        if not os.path.exists(tmp_inp):
            os.chdir(orig_dir)
            return False
        with open(tmp_inp, 'r') as f:
            lines = f.readlines()
        def make_adaptive_static(lines, allsdtol='0.05', continue_yes=True):
            new_lines = []
            for line in lines:
                s = line.strip()
                if s.lower().startswith('*static'):
                    if 'stabilize' not in s.lower():
                        s = s + ', stabilize'
                    parts = [p.strip() for p in s.split(',')]
                    kept = []
                    for p in parts:
                        pl = p.lower()
                        if pl.startswith('*static'):
                            kept.append('*Static')
                        elif pl.startswith('allsdtol'):
                            continue
                        elif pl.startswith('continue'):
                            continue
                        else:
                            kept.append(p)
                    kept.append('ALLSDTOL={}'.format(allsdtol))
                    kept.append('CONTINUE={}'.format('YES' if continue_yes else 'NO'))
                    line = ', '.join(kept) + '\n'
                new_lines.append(line)
            return new_lines
        patched = make_adaptive_static(lines, allsdtol='0.05', continue_yes=True)
        with open(sim_inp, 'w') as f:
            f.writelines(patched)
        if tmp_name in mdb.jobs:
            del mdb.jobs[tmp_name]
        final_name = 'JobSim_' + sid + '_' + timestamp
        mdb.JobFromInputFile(name=final_name, inputFileName=os.path.abspath(sim_inp),
                             scratch=os.path.abspath(out_dir))
        cae_path = os.path.join(out_dir, safe_name + '_simulation.cae')
        mdb.saveAs(pathName=cae_path)
        os.chdir(orig_dir)
        return True
    except Exception as e:
        print("Failed to finalize input/job: {}".format(str(e)))
        try:
            os.chdir(orig_dir)
        except:
            pass
        return False

def main():
    print("Main function started")
    RES = r'D:\CODE\Abaqus_batch_test\Abaqus_Results'
    lg = logger(RES)
    sim_sum = []
    folders = [os.path.join(RES, d) for d in os.listdir(RES) if os.path.isdir(os.path.join(RES, d))]
    print("Found {} folders".format(len(folders)))
    for folder in folders:
        base_files = glob.glob(os.path.join(folder, '*_base.inp'))
        if not base_files:
            continue
        base_inp = base_files[0]
        base_no_ext = os.path.splitext(os.path.basename(base_inp))[0]
        safe = base_no_ext[:-5] if base_no_ext.endswith('_base') else base_no_ext
        sim_inp = os.path.join(folder, safe + '_simulation.inp')
        cae_file = os.path.join(folder, safe + '.cae')
        if os.path.exists(sim_inp):
            sim_sum.append({'sample': os.path.basename(folder), 'status': 'SKIPPED', 'simulation_inp': sim_inp})
            print("Sample {} skipped (exists)".format(folder))
            continue
        ok = create_simulation(cae_file, base_inp, sim_inp, os.path.basename(folder))
        status = 'SUCCESS' if ok else 'FAILED'
        sim_sum.append({'sample': os.path.basename(folder), 'status': status, 'simulation_inp': sim_inp if ok else None})
        print("Sample {}: {}".format(folder, status))
        try:
            mdb.close()
            Mdb()
            gc.collect()
        except Exception as e:
            print("MDB reset failed: {}".format(str(e)))
    summary_file = os.path.join(RES, 'simulation_setup_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(sim_sum, f, indent=4)
    print("\nSimulation setup completed")
    print("Summary saved to: {}".format(summary_file))
    successful = len([s for s in sim_sum if s['status'] == 'SUCCESS'])
    print("Total: {}, Successful: {}, Failed: {}".format(len(sim_sum), successful, len(sim_sum) - successful))


if __name__ == '__main__':
    main()
#abaqus cae noGUI=simulation_setup.py