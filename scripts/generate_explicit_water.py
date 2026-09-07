"""Seeded, offline rigid TIP4P-Ew heating trajectory (OpenMM)."""
import json
from pathlib import Path
import numpy as np
import openmm as mm
from openmm import app, unit

ICE_CELL = 6.35
BOX = ICE_CELL * 2
SEED = 22039
ROOT = Path(__file__).resolve().parents[1]
PLATFORM = mm.Platform.getPlatformByName('CPU')
PLATFORM_PROPERTIES = {'Threads': '2', 'DeterministicForces': 'true'}


def minimum_image(d, box=BOX):
    return d - box * np.rint(d / box)


def ice_configuration(cells=2, seed=SEED):
    """Proton-disordered ice Ic for 8 * cells**3 molecules, satisfying the ice rules."""
    count = 8 * cells ** 3
    box = ICE_CELL * cells
    rng = np.random.default_rng(seed)
    fcc = np.array([[0,0,0], [0,.5,.5], [.5,0,.5], [.5,.5,0]])
    basis = np.concatenate([fcc, fcc + .25])
    lattice = np.indices((cells,)*3).reshape(3,-1).T
    oxygen = ((lattice[:,None,:] + basis).reshape(-1,3) + .125) * ICE_CELL
    d = minimum_image(oxygen[None] - oxygen[:,None], box)
    distance = np.linalg.norm(d, axis=-1)
    adjacent = (distance < 3) & (distance > 0)
    graph = [set(np.flatnonzero(row)) for row in adjacent]
    stack, circuit = [0], []
    while stack:
        i = stack[-1]
        if graph[i]:
            j = int(rng.choice(sorted(graph[i])))
            graph[i].remove(j)
            graph[j].remove(i)
            stack.append(j)
        else:
            circuit.append(stack.pop())
    directed = np.zeros((count,count), bool)
    for i, j in zip(circuit[:-1], circuit[1:]):
        directed[i,j] = True
    xyz = np.zeros((count,3,3))
    xyz[:,0] = oxygen
    theta = np.deg2rad(104.52/2)
    for i in range(count):
        v = d[i,np.flatnonzero(directed[i])]
        v /= np.linalg.norm(v,axis=1)[:,None]
        bisector = v[0]+v[1]
        bisector /= np.linalg.norm(bisector)
        tangent = v[0]-v[1]
        tangent /= np.linalg.norm(tangent)
        xyz[i,1:] = oxygen[i] + .9572 * np.array([
            np.cos(theta)*bisector + np.sin(theta)*tangent,
            np.cos(theta)*bisector - np.sin(theta)*tangent])
    return xyz, directed, adjacent


def tetrahedral_order(oxygen):
    d = minimum_image(oxygen[None]-oxygen[:,None])
    r2 = np.sum(d*d, axis=-1)
    np.fill_diagonal(r2, np.inf)
    nearest = np.argsort(r2,axis=1)[:,:4]
    v = d[np.arange(64)[:,None], nearest]
    v /= np.linalg.norm(v,axis=-1)[...,None]
    q = np.ones(64)
    for j in range(4):
        for k in range(j+1,4):
            q -= 3/8*(np.sum(v[:,j]*v[:,k],axis=-1)+1/3)**2
    return float(q.mean())


def hydrogen_bonds(xyz):
    oxygen = xyz[:,0]
    oo = minimum_image(oxygen[None] - oxygen[:,None])
    distance = np.linalg.norm(oo,axis=-1)
    bonds = set()
    for h in (1,2):
        oh = xyz[:,h]-oxygen
        ha = oo-oh[:,None]
        norm = np.linalg.norm(ha,axis=-1)*np.linalg.norm(oh,axis=-1)[:,None]
        cosine = np.sum(-oh[:,None]*ha,axis=-1)/np.maximum(norm,1e-12)
        valid = (distance <= 3.5) & (distance > 0) & (cosine <= np.cos(np.deg2rad(150)))
        bonds.update((int(i),h,int(j)) for i,j in np.argwhere(valid))
    return bonds


def build_system(xyz, box):
    """Force field, virtual sites and PME setup shared by generate() and minimized()."""
    topology = app.Topology()
    chain = topology.addChain()
    topology.setPeriodicBoxVectors(np.eye(3)*box/10*unit.nanometer)
    for _ in xyz:
        residue = topology.addResidue('HOH',chain)
        atoms = [topology.addAtom(name, element, residue) for name,element in
                 [('O',app.element.oxygen),('H1',app.element.hydrogen),('H2',app.element.hydrogen)]]
        topology.addBond(atoms[0],atoms[1]); topology.addBond(atoms[0],atoms[2])
    forcefield = app.ForceField('tip4pew.xml')
    modeller = app.Modeller(topology,xyz.reshape(-1,3)/10*unit.nanometer)
    modeller.addExtraParticles(forcefield)
    system = forcefield.createSystem(modeller.topology,nonbondedMethod=app.PME,
        nonbondedCutoff=.6*unit.nanometer,rigidWater=True,ewaldErrorTolerance=1e-5)
    return modeller.topology, system, modeller.positions


def minimized(cells):
    """Energy-minimized ice configuration for the browser, in the same units as water.bin."""
    xyz, _, _ = ice_configuration(cells=cells)
    box = ICE_CELL * cells
    topology, system, positions = build_system(xyz, box)
    integrator = mm.LangevinMiddleIntegrator(180*unit.kelvin, 1/unit.picosecond, 2*unit.femtosecond)
    simulation = app.Simulation(topology, system, integrator, PLATFORM, PLATFORM_PROPERTIES)
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy()
    state = simulation.context.getState(getPositions=True)
    drawn = np.array(state.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
    drawn = drawn.reshape(-1, 4, 3)[:, :3]      # drop the virtual site
    drawn[:, 1:] -= ICE_CELL * cells * np.rint((drawn[:, 1:] - drawn[:, :1]) / box)
    drawn[:, 0] -= box * np.floor(drawn[:, 0] / box)
    return drawn


def export_initial_states():
    for cells in (2, 3, 4):
        drawn = minimized(cells)
        destination = ROOT / f'public/data/ice-{8 * cells ** 3}.bin'
        drawn.astype('<f4').tofile(destination)
        print(f'Wrote {destination.name}: {drawn.shape[0]} molecules')


def generate():
    xyz, _, _ = ice_configuration()
    topology, system, positions = build_system(xyz, BOX)
    integrator = mm.LangevinMiddleIntegrator(180*unit.kelvin,1/unit.picosecond,.002*unit.picoseconds)
    integrator.setRandomNumberSeed(SEED)
    integrator.setConstraintTolerance(1e-7)
    simulation = app.Simulation(topology,system,integrator,PLATFORM,PLATFORM_PROPERTIES)
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy(tolerance=1*unit.kilojoule_per_mole/unit.nanometer,maxIterations=1000)
    simulation.context.setVelocitiesToTemperature(180*unit.kelvin,SEED)
    simulation.step(5000) # 10 ps unrecorded equilibration
    print('Equilibrated 10 ps',flush=True)
    physical = [a.index for a in topology.atoms() if a.element is not None]
    frames, coordinates = [], []
    dof = 3*192-system.getNumConstraints()-3
    for frame in range(2001):
        time = frame*.02
        temperature = float(180+270*np.clip((time-5)/20,0,1))
        integrator.setTemperature(temperature*unit.kelvin)
        if frame:
            simulation.step(10)
        state = simulation.context.getState(getPositions=True,getVelocities=True,getEnergy=True)
        x = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)[physical].reshape(64,3,3)
        local = minimum_image(x[:,1:]-x[:,:1])
        x[:,0] %= BOX
        x[:,1:] = x[:,:1]+local
        velocity = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)[physical].reshape(64,3,3)
        ke = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        if not np.isfinite(x).all() or not np.isfinite(energy):
            raise RuntimeError('Nonfinite simulation')
        frames.append({'timePs':round(time,2),'temperature':round(temperature,2),
            'kineticTemperature':round(2*ke/(dof*.008314462618),3),
            'oxygenRmsSpeedAngstromPerPs':round(float(np.sqrt(np.mean(np.sum(velocity[:,0]**2,axis=-1)))*10),4),
            'tetrahedralOrder':round(tetrahedral_order(x[:,0]),5),
            'hydrogenBondCount':len(hydrogen_bonds(x)),
            'potentialEnergyKJPerMol':round(energy,4)})
        coordinates.append(x.copy())
        if frame%250==0:
            print(frames[-1],flush=True)
    metadata = {'version':2,'particles':64,'box':BOX,'units':'angstrom','model':'TIP4P-Ew',
        'atomOrder':['O','H','H'],'coordinateFile':'water.bin','coordinateType':'float32-le',
        'citation':'https://doi.org/10.1063/1.1683075',
        'protocol':'Proton-disordered ice Ic; energy minimization and 10 ps equilibration at 180 K; fixed-volume periodic Langevin dynamics; recorded 5 ps at 180 K, 20 ps ramp to 450 K, 15 ps at 450 K. Rapid heating of a small system, not an equilibrium melting-point measurement.',
        'seed':SEED,'timestepFs':2,'frictionPerPs':1,'sampleStride':10,'equilibrationPs':10,
        'nonbondedMethod':'PME','cutoffAngstrom':6,'ewaldErrorTolerance':1e-5,
        'openmmVersion':mm.__version__,'platform':PLATFORM.getName(),'constraints':'rigid water; virtual charge site omitted from display',
        'hydrogenBondCriterion':{'ooMaxAngstrom':3.5,'ohaMinDegrees':150},'frames':frames}
    array = np.asarray(coordinates,dtype='<f4')
    from test_explicit_water import validate_trajectory
    print(validate_trajectory(metadata,array),flush=True)
    destination = ROOT / 'reference'
    destination.mkdir(exist_ok=True)
    array.tofile(destination/'water.bin')
    (destination/'water.json').write_text(json.dumps(metadata,separators=(',',':'))+'\n')


if __name__ == '__main__':
    export_initial_states()
    generate()
