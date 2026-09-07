"""Geometry, ice-rule, continuity and thermal checks for the offline sample."""
import json
import unittest
import numpy as np
from generate_explicit_water import BOX, ROOT, ice_configuration, minimum_image, hydrogen_bonds, tetrahedral_order


def validate_trajectory(metadata, xyz):
    assert xyz.shape == (len(metadata['frames']),64,3,3)
    assert np.isfinite(xyz).all()
    assert np.all((xyz[:,:,0]>=0)&(xyz[:,:,0]<BOX))
    oh = xyz[:,:,1:]-xyz[:,:,:1]
    lengths = np.linalg.norm(oh,axis=-1)
    assert np.max(np.abs(lengths-.9572)) < 2e-4
    angles = np.rad2deg(np.arccos(np.clip(np.sum(oh[:,:,0]*oh[:,:,1],axis=-1)/np.prod(lengths,axis=-1),-1,1)))
    assert np.max(np.abs(angles-104.52)) < .01
    jumps = np.linalg.norm(minimum_image(np.diff(xyz[:,:,0],axis=0)),axis=-1)
    assert jumps.max() < 1.5
    assert np.allclose(np.diff([f['timePs'] for f in metadata['frames']]),.02)
    for i in [0,len(xyz)//2,len(xyz)-1]:
        assert len(hydrogen_bonds(xyz[i])) == metadata['frames'][i]['hydrogenBondCount']
    series = [hydrogen_bonds(x) for x in xyz]
    broken = sum(len(a-b) for a,b in zip(series,series[1:]))
    formed = sum(len(b-a) for a,b in zip(series,series[1:]))
    seen, lost, reformed = set(series[0]), set(), set()
    for previous,current in zip(series,series[1:]):
        lost |= previous-current
        reformed |= (current-previous)&lost
        seen |= current
    low = metadata['frames'][:251]
    high = metadata['frames'][-501:]
    summary = {'maxOxygenStepAngstrom':float(jumps.max()),'brokenEvents':broken,'formedEvents':formed,'distinctReformedBonds':len(reformed),'distinctBonds':len(seen)}
    for name,frames in [('low',low),('high',high)]:
        summary[name] = {key:float(np.mean([f[key] for f in frames])) for key in ['kineticTemperature','oxygenRmsSpeedAngstromPerPs','tetrahedralOrder','hydrogenBondCount']}
    increments = minimum_image(np.diff(xyz[:,:,0],axis=0))
    for name,start,end in [('cold5Ps',0,250),('hot10Ps',1500,2000)]:
        displacement = increments[start:end].sum(axis=0)
        displacement -= displacement.mean(axis=0)
        summary[name+'MsdAngstrom2'] = float(np.mean(np.sum(displacement**2,axis=-1)))
    assert summary['high']['kineticTemperature'] > summary['low']['kineticTemperature']*1.5
    assert summary['low']['tetrahedralOrder'] > .8
    assert summary['high']['tetrahedralOrder'] < .8
    assert summary['cold5PsMsdAngstrom2'] < 1
    assert summary['hot10PsMsdAngstrom2'] > 5
    assert broken > 100 and formed > 100 and reformed
    return summary


class ExplicitWaterTests(unittest.TestCase):
    def test_ice_rules(self):
        xyz,directed,adjacent = ice_configuration()
        self.assertTrue(np.all(adjacent.sum(axis=1)==4))
        self.assertTrue(np.all(directed.sum(axis=1)==2))
        self.assertTrue(np.all(directed.sum(axis=0)==2))
        self.assertTrue(np.array_equal(directed|directed.T,adjacent))
        self.assertFalse(np.any(directed&directed.T))
        self.assertAlmostEqual(tetrahedral_order(xyz[:,0]),1)
        self.assertEqual(len(hydrogen_bonds(xyz)),128)

    def test_generated_trajectory(self):
        metadata = json.loads((ROOT/'reference/water.json').read_text())
        xyz = np.fromfile(ROOT/'reference/water.bin',dtype='<f4').reshape(-1,64,3,3)
        print(json.dumps(validate_trajectory(metadata,xyz),indent=2))


class InitialStates(unittest.TestCase):
    def test_ice_rules_hold_for_every_exported_cell_count(self):
        for cells in (2, 3, 4):
            count = 8 * cells ** 3
            xyz, directed, adjacent = ice_configuration(cells=cells)
            self.assertEqual(xyz.shape, (count, 3, 3))
            self.assertTrue(np.all(adjacent.sum(axis=1) == 4))
            # Each oxygen donates two protons and accepts two.
            self.assertTrue(np.all(directed.sum(axis=1) == 2))
            self.assertTrue(np.all(directed.sum(axis=0) == 2))
            self.assertFalse(np.any(directed & directed.T))

    def test_exported_files_are_rigid_wrapped_and_the_right_size(self):
        for cells in (2, 3, 4):
            count = 8 * cells ** 3
            box = 6.35 * cells
            path = ROOT / f'public/data/ice-{count}.bin'
            raw = np.fromfile(path, dtype='<f4')
            self.assertEqual(raw.size, count * 9)
            xyz = raw.reshape(count, 3, 3).astype(float)
            self.assertTrue(np.isfinite(xyz).all())
            self.assertTrue(np.all((xyz[:, 0] >= 0) & (xyz[:, 0] < box)))
            oh = xyz[:, 1:] - xyz[:, :1]
            lengths = np.linalg.norm(oh, axis=-1)
            self.assertLess(np.max(np.abs(lengths - .9572)), 2e-3)
            cosines = np.sum(oh[:, 0] * oh[:, 1], axis=-1) / np.prod(lengths, axis=-1)
            angles = np.rad2deg(np.arccos(np.clip(cosines, -1, 1)))
            self.assertLess(np.max(np.abs(angles - 104.52)), .1)
            # Nearest oxygen neighbours stay at the lattice spacing after minimisation.
            d = minimum_image(xyz[:, 0][None] - xyz[:, 0][:, None], box)
            r = np.linalg.norm(d, axis=-1)
            np.fill_diagonal(r, np.inf)
            self.assertGreater(r.min(), 2.3)


if __name__ == '__main__':
    unittest.main()
