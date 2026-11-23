"""Checks for paired scientific evidence accounting and negative controls."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from audit_evidence import aggregate_pairs, audit, main, perfect_drafter_controls, render_markdown


def row(method, tv, seed=17, sequence=1, regime='smooth'):
    return {'method':method,'tv':tv,'model_seed':seed,'sequence_seed':sequence,
            'regime':regime,'committed':256}


class EvidenceAuditTests(unittest.TestCase):
    def test_paired_delta_acceptance_identity_and_model_aggregation(self):
        rows = [row('instant',0.1),row('prospective',0.2),
                row('instant',0.3,sequence=2),row('prospective',0.2,sequence=2),
                row('instant',0.2,seed=23),row('prospective',0.2,seed=23),
                row('leaky',0.8)]
        result = aggregate_pairs(rows)
        overall = result['overall']
        self.assertEqual(overall['paired_contexts'],3)
        self.assertEqual(overall['model_seeds'],2)
        self.assertEqual(overall['prospective_fidelity_wins'],1)
        self.assertEqual(overall['prospective_fidelity_losses'],1)
        self.assertEqual(overall['fidelity_ties'],1)
        self.assertAlmostEqual(overall['mean_tv_prospective_minus_instant'],0)
        self.assertAlmostEqual(overall['mean_instant_expected_single_proposal_acceptance'],0.8)
        self.assertAlmostEqual(overall['mean_prospective_expected_single_proposal_acceptance'],0.8)
        self.assertEqual(len(result['by_model_and_regime']),2)
        for pair in result['paired_contexts']:
            self.assertAlmostEqual(pair['expected_acceptance_prospective_minus_instant'],
                                   pair['instant_tv']-pair['prospective_tv'])

    def test_pair_order_does_not_change_evidence(self):
        rows = [row('instant',0.1),row('prospective',0.2),
                row('instant',0.2,regime='abrupt'),row('prospective',0.25,regime='abrupt')]
        self.assertEqual(aggregate_pairs(rows),aggregate_pairs(list(reversed(rows))))

    def test_missing_pairs_duplicates_and_count_mismatches_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            aggregate_pairs([row('instant',0.1)])
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            aggregate_pairs([row('instant',0.1),row('instant',0.1),row('prospective',0.2)])
        with self.assertRaisesRegex(ValueError,'different committed-token'):
            aggregate_pairs([row('instant',0.1),dict(row('prospective',0.2),committed=255)])
        with self.assertRaisesRegex(ValueError,'No instantaneous'):
            aggregate_pairs([row('leaky',0.2)])

    def test_invalid_tv_and_tolerances_fail(self):
        for value in (float('nan'),float('inf'),-0.1,1.1,True,'0.1'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                aggregate_pairs([row('instant',value),row('prospective',0.2)])
        for tolerance in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                aggregate_pairs([row('instant',0.1),row('prospective',0.2)],tolerance)

    def test_tolerance_marks_numerical_ties(self):
        result = aggregate_pairs([row('instant',0.1),row('prospective',0.1+1e-13)],tolerance=1e-12)
        self.assertEqual(result['overall']['fidelity_ties'],1)
        self.assertEqual(result['paired_contexts'][0]['outcome'],'tie')

    def test_perfect_drafter_controls_expose_finite_filter_error(self):
        result = perfect_drafter_controls(steps=128)
        self.assertEqual(len(result['results']),3)
        for scenario in result['results']:
            with self.subTest(scenario=scenario['scenario']):
                methods = {entry['method']:entry for entry in scenario['methods']}
                self.assertTrue(scenario['guards']['passed'])
                self.assertEqual(methods['instant']['mean_tv'],0)
                self.assertEqual(methods['ideal']['mean_tv'],0)
                self.assertEqual(methods['instant']['expected_single_proposal_acceptance'],1)
                for item in methods.values():
                    self.assertAlmostEqual(item['expected_single_proposal_acceptance'],1-item['mean_tv'])
                if scenario['scenario']=='constant':
                    self.assertTrue(all(item['mean_tv']<1e-14 for item in methods.values()))
                else:
                    self.assertGreater(methods['prospective']['mean_tv'],1e-4)
                    self.assertLess(methods['prospective']['mean_tv'],methods['leaky']['mean_tv'])

    def test_control_inputs_are_validated(self):
        for steps in (0,1,True,1.5):
            with self.assertRaises(ValueError):
                perfect_drafter_controls(steps=steps)

    def test_audit_writes_companion_files_without_modifying_source(self):
        source = {'metadata':{'device':'cpu','args':{'tau':2.,'tau_a':0.2}},
                  'runs':[row('instant',0.1),row('prospective',0.2)]}
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            path=directory/'results.json'
            path.write_text(json.dumps(source))
            before=path.read_bytes()
            result=audit(directory)
            self.assertEqual(result['source']['sha256'],hashlib.sha256(before).hexdigest())
            self.assertEqual(result['correctness_controls_status'],'PASS')
            self.assertEqual(result['discovery_status'],'NOT_ESTABLISHED')
            self.assertIn('worse in every paired context',result['finding'])
            self.assertIn('Correctness controls: PASS. Discovery evidence: not established.',render_markdown(result))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--results',str(directory)]),0)
            self.assertEqual(path.read_bytes(),before)
            self.assertTrue((directory/'audit_results.json').is_file())
            self.assertTrue((directory/'AUDIT_EVIDENCE.md').is_file())
            written=json.loads((directory/'audit_results.json').read_text())
            self.assertEqual(written['paired_saved_evidence']['overall']['prospective_fidelity_losses'],1)

    def test_gpu_timing_file_cannot_be_mislabeled_as_cpu_fidelity(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            (directory/'results.json').write_text(json.dumps({'metadata':{'device':'cuda'}}))
            with self.assertRaisesRegex(ValueError,'saved CPU'):
                audit(directory)


if __name__=='__main__':
    unittest.main()
