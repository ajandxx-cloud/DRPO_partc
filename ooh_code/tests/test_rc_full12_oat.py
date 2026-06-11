import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import sensitivity_analysis_dspo_plus_spo_oat as oat


class TestRcFull12OAT(unittest.TestCase):
    def test_rc_full12_profile_shape(self):
        self.assertEqual(len(oat.RC_FULL12_FACTORS), 12)
        for factor, levels in oat.RC_FULL12_FACTORS.items():
            self.assertEqual(len(levels), 5, msg=f"{factor} must have 5 levels")
            self.assertIn(factor, oat.RC_FULL12_DEFAULT_CONFIG)
            dtype = oat.factor_dtype(factor)
            if dtype == "int":
                self.assertTrue(all(float(v).is_integer() for v in levels))

    def test_expected_stage_job_counts(self):
        args = SimpleNamespace(
            factor_grid=oat.RC_FULL12_FACTORS,
            default_config=oat.RC_FULL12_DEFAULT_CONFIG,
            stage1_seeds=[0, 21, 42, 63, 84],
            stage2_seeds=[0, 7, 14, 21, 28, 35, 42, 49, 56, 63],
        )
        stage1_jobs = oat.build_stage1_jobs(args)
        self.assertEqual(len(stage1_jobs), 12 * 5 * 5)

        candidates = {factor: float(levels[0]) for factor, levels in oat.RC_FULL12_FACTORS.items()}
        s2vals = oat.stage2_values(candidates, args.factor_grid, args.default_config)
        stage2_jobs = oat.build_stage2_jobs(args, s2vals)
        self.assertEqual(len(stage2_jobs), 12 * 2 * 10)

    def test_metric_parsing_served_rate(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "logfile.log"
            log.write_text(
                "\n".join(
                    [
                        "Net profit: 1234.5",
                        "total costs: 432.1",
                        "Quit rate: 12.5%",
                        "Accepted customers: 1600",
                        "Total customers: 2000",
                    ]
                ),
                encoding="utf-8",
            )
            metrics = oat.parse_metrics(log)
            self.assertIsNotNone(metrics)
            self.assertAlmostEqual(metrics["net_profit"], 1234.5)
            self.assertAlmostEqual(metrics["total_costs"], 432.1)
            self.assertAlmostEqual(metrics["quit_rate"], 12.5)
            self.assertAlmostEqual(metrics["served_rate"], 0.8)

    def test_guardrail_filter_and_fallback(self):
        stage2_summary = [
            {"factor": "outside_option_util", "value": -1.0, "net_profit_mean": 100.0, "quit_rate_mean": 4.0, "served_rate_mean": 0.95},
            {"factor": "outside_option_util", "value": -2.0, "net_profit_mean": 130.0, "quit_rate_mean": 8.5, "served_rate_mean": 0.88},
            {"factor": "k", "value": 10.0, "net_profit_mean": 100.0, "quit_rate_mean": 4.0, "served_rate_mean": 0.95},
            {"factor": "k", "value": 5.0, "net_profit_mean": 110.0, "quit_rate_mean": 5.0, "served_rate_mean": 0.94},
        ]
        candidates = {"outside_option_util": -2.0, "k": 5.0}
        factor_grid = {"outside_option_util": [-2.0, -1.0], "k": [5.0, 10.0]}
        default_config = {"outside_option_util": -1.0, "k": 10.0}

        ranking, recs = oat.compute_stage2_guardrail_and_recommendations(
            stage2_summary_enhanced=stage2_summary,
            candidates=candidates,
            factor_grid=factor_grid,
            default_config=default_config,
            primary_metric="net_profit",
            guardrail_quit_delta_pp=2.0,
            guardrail_served_rate_delta=-0.02,
        )

        rec_map = {r["factor"]: r for r in recs}
        self.assertEqual(rec_map["outside_option_util"]["recommendation_type"], "fallback_risky")
        self.assertEqual(rec_map["outside_option_util"]["risk_flag"], "RED")
        self.assertEqual(rec_map["k"]["recommendation_type"], "guardrail_pass")
        self.assertEqual(rec_map["k"]["risk_flag"], "")

        self.assertEqual(len(ranking), 2)
        self.assertIn("rank_by_primary_gain", ranking[0])

    def test_persist_small_e2e_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            args = SimpleNamespace(
                factor_grid={"outside_option_util": [-2.0, -1.0]},
                default_config={"outside_option_util": -1.0},
                primary_metric="net_profit",
                guardrail_quit_delta_pp=2.0,
                guardrail_served_rate_delta=-0.02,
            )
            catalog = oat.build_parameter_catalog("rc_full12", args.factor_grid, args.default_config)
            s1 = [
                oat.RunRecord("stage1", "outside_option_util", -1.0, 0, 80, "a", "completed", 1.0, 100.0, 50.0, 4.0, 950.0, 1000.0, 0.95, "l1", "cmd"),
                oat.RunRecord("stage1", "outside_option_util", -2.0, 0, 80, "b", "completed", 1.0, 120.0, 55.0, 3.0, 980.0, 1000.0, 0.98, "l2", "cmd"),
            ]
            s2 = [
                oat.RunRecord("stage2", "outside_option_util", -1.0, 0, 200, "c", "completed", 1.0, 102.0, 49.0, 4.0, 950.0, 1000.0, 0.95, "l3", "cmd"),
                oat.RunRecord("stage2", "outside_option_util", -2.0, 0, 200, "d", "completed", 1.0, 118.0, 52.0, 5.5, 940.0, 1000.0, 0.94, "l4", "cmd"),
            ]
            candidates = {"outside_option_util": -2.0}

            oat.persist(out, args, catalog, s1, s2, candidates)

            expected = [
                out / "parameter_catalog.csv",
                out / "stage1_summary_enhanced.csv",
                out / "stage2_guardrail_ranking.csv",
                out / "final_recommendations.csv",
                out / "sensitivity_scores.csv",
            ]
            for p in expected:
                self.assertTrue(p.exists(), msg=f"Missing expected output: {p}")
                self.assertGreater(p.stat().st_size, 0, msg=f"Empty output: {p}")


if __name__ == "__main__":
    unittest.main()
