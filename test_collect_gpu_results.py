"""Pure file/schema tests; fixture values are not measured GPU results."""
import json
from pathlib import Path
import tempfile
import unittest

from collect_gpu_results import collect, write_summary


def fixture(k=4, hardware='fixture-device', status='complete'):
    return {'status':status, 'metadata':{
        'device':'cuda', 'hardware':hardware, 'benchmark_executed':True,
        'torch_version':'fixture', 'numpy_version':'fixture', 'teacher_width':32,
        'tokens':8, 'sequences':1, 'repeats':1, 'seeds':[17], 'k':k},
        'summary':[{'regime':'smooth','method':'prospective','runs':1,
                    'ms_per_token_median':2.0,'tokens_per_verification':2.0,
                    'paired_speedup_vs_instant_median':0.9}]}


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)

    def put(self, directory, payload):
        dest=self.root/directory
        dest.mkdir()
        (dest/'gpu_results.json').write_text(json.dumps(payload))

    def test_completed_results_summarize_and_partial_is_labelled(self):
        self.put('k4',fixture())
        partial=self.root/'k8'; partial.mkdir()
        (partial/'gpu_checkpoint.json').write_text('{"status":"incomplete"}')
        rows,info,incomplete=collect(self.root)
        self.assertEqual(incomplete,['k8'])
        write_summary(self.root,rows,info,incomplete)
        self.assertIn('partial sweep',(self.root/'SUMMARY.md').read_text())
        self.assertIn('0.900',(self.root/'SUMMARY.md').read_text())

    def test_no_completed_results_cannot_become_summary(self):
        with self.assertRaisesRegex(ValueError,'No completed'):
            collect(self.root)

    def test_incomplete_final_filename_is_rejected(self):
        self.put('k4',fixture(status='incomplete'))
        with self.assertRaisesRegex(ValueError,'not marked complete'):
            collect(self.root)

    def test_mixed_hardware_cannot_be_pooled(self):
        self.put('k4',fixture())
        self.put('k8',fixture(k=8,hardware='another-fixture-device'))
        with self.assertRaisesRegex(ValueError,'different devices'):
            collect(self.root)

    def test_duplicate_draft_lengths_cannot_be_pooled(self):
        self.put('first',fixture())
        self.put('second',fixture())
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            collect(self.root)

    def test_different_model_provenance_cannot_be_pooled(self):
        first=fixture()
        first['metadata']['source_provenance']={'asset_manifest_sha256':'first-fixture'}
        second=fixture(k=8)
        second['metadata']['source_provenance']={'asset_manifest_sha256':'second-fixture'}
        self.put('k4',first)
        self.put('k8',second)
        with self.assertRaisesRegex(ValueError,'different devices'):
            collect(self.root)


if __name__=='__main__':
    unittest.main()
