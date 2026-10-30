import unittest
from fusion_multimodal import freshness_weight, fuse

class FusionTests(unittest.TestCase):
    def test_fresh_radar_has_full_weight(self):
        self.assertEqual(freshness_weight(10), 1.0)
    def test_stale_radar_has_no_weight(self):
        self.assertEqual(freshness_weight(120), 0.0)
    def test_fusion_never_claims_probability(self):
        out=fuse({'radar':{'radar_age':10,'nucleo_principal':{'dbz':{'p99':65}}},
                  'rma10':{'predicciones':[{'distance_km':20,'horizontes':{'10':{'strengthened':.8,'toward_bahia':.9}}}]},
                  'environment':{'impacto':.7},'goes':{'goes19_conveccion':.8}})
        self.assertTrue(0 <= out['evidence_score'] <= 1)
        self.assertFalse(out['semantics']['evidence_score_is_calibrated_probability'])
        self.assertGreater(out['confidence'], .3)

if __name__ == '__main__': unittest.main()
