from copy import deepcopy
import json
import unittest
from TemplatePattern_final.shared.io import ROOT
from TemplatePattern_final.shared.template_selection import select_template
from TemplatePattern_final.sim.spec_converter import GarmentSpecConverter
from TemplatePattern_final.sim.tpose_placement import BodyPosePlacement


class RestoredContractTest(unittest.TestCase):
    def test_original_sewing_and_geometry_are_exact(self):
        references = json.loads((ROOT/'assets/placement_reference.json').read_text())
        for style in ('short_sleeve','long_sleeve'):
            for pose in ('a30','a45','a60','tpose'):
                spec, debug = GarmentSpecConverter().convert({'style':style}, f'{style}_{pose}')
                original = json.loads((ROOT/debug['reference_override']['path']).read_text())
                self.assertEqual(spec, original)
                key = style+(':tpose' if style=='short_sleeve' or pose=='tpose' else ':a45')
                body = deepcopy(references[key])
                BodyPosePlacement().apply(spec,style,body,pose,30,2)
                self.assertEqual(spec['pattern']['stitches'], original['pattern']['stitches'])
                for name, panel in spec['pattern']['panels'].items():
                    for field in ('vertices','edges','rotation'):
                        self.assertEqual(panel[field],original['pattern']['panels'][name][field])
                    for a,b in zip(panel['translation'],original['pattern']['panels'][name]['translation']):
                        self.assertAlmostEqual(a,b,places=5)

    def test_body_changes_positions_not_sewing(self):
        references = json.loads((ROOT/'assets/placement_reference.json').read_text())
        for style in ('short_sleeve','long_sleeve'):
            spec,_ = GarmentSpecConverter().convert({'style':style},f'{style}_a30')
            original = deepcopy(spec)
            body = deepcopy(references[style+(':tpose' if style=='short_sleeve' else ':a45')])
            def scale(value):
                if isinstance(value,dict):
                    return {k:(v if k.endswith('_deg') else scale(v)) for k,v in value.items()}
                if isinstance(value,list):return [scale(v) for v in value]
                return value*1.15
            body=scale(body)
            BodyPosePlacement().apply(spec,style,body,'a30',30,2)
            self.assertEqual(spec['pattern']['stitches'],original['pattern']['stitches'])
            for name,panel in spec['pattern']['panels'].items():
                self.assertNotEqual(panel['translation'],original['pattern']['panels'][name]['translation'])
                for field in ('vertices','edges'):
                    self.assertEqual(panel[field],original['pattern']['panels'][name][field])

    def test_capacity_and_height_fallback(self):
        refs=json.loads((ROOT/'config/size_body_reference_v4.json').read_text())
        for height in range(1500,2001):
            for waist in (640,780,1040,1200):
                body={k:refs['sizes']['M'][k] for k in ('height','chest','waist','hip','shoulder_width','arm_length')}
                body.update(height=height,waist=waist)
                result=select_template(body,refs)
                self.assertIn(result['nearest_size'],refs['sizes'])
                if result['selection_strategy']=='capacity_covering':
                    self.assertTrue(all(result['selected_capacity_mm'][k]>=v for k,v in body.items()))
        body={k:refs['sizes']['M'][k] for k in ('height','chest','waist','hip','shoulder_width','arm_length')}
        body['waist']=900
        self.assertEqual(select_template(body,refs)['nearest_size'],'XXL')
        body['waist']=1200
        self.assertEqual(select_template(body,refs)['nearest_size'],'XXXL')


if __name__=='__main__':unittest.main()
