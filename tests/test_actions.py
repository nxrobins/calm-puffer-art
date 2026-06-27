import unittest

from calm_puffer_art.actions import (
    ACTION_SPACE_STATE_KEY,
    AdaptiveActionSpace,
    ChunkActionCodec,
    CommandActionCodec,
    LatentPatchActionCodec,
    ReasoningStepCodec,
    TokenActionCodec,
    action_codec_key,
    action_logprob_stats,
    action_space_checkpoint_metadata,
    action_space_signature,
    semantic_bandwidth,
)
from calm_puffer_art.types import ActionUnit


class ActionCodecTests(unittest.TestCase):
    def test_chunk_codec_round_trips_with_higher_bandwidth_than_tokens(self):
        text = "alpha beta gamma delta epsilon"
        token_actions = TokenActionCodec().encode(text)
        chunk_actions = ChunkActionCodec(chunk_size=2).encode(text)

        self.assertEqual(ChunkActionCodec(chunk_size=2).decode(chunk_actions), text)
        self.assertEqual(len(token_actions), 5)
        self.assertEqual(len(chunk_actions), 3)
        self.assertGreater(semantic_bandwidth(chunk_actions), semantic_bandwidth(token_actions))

    def test_action_space_signature_tracks_active_codec_ladder(self):
        action_space = AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4)
        base_signature = action_space_signature(action_space)

        action_space.add_codec(ChunkActionCodec(chunk_size=4))
        expanded_signature = action_space_signature(action_space)
        restored = AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4)
        restored.load_state_dict(action_space.state_dict())

        self.assertIsNotNone(base_signature)
        self.assertIn("chunk_chunk_size_2", base_signature)
        self.assertNotEqual(base_signature, expanded_signature)
        self.assertIn("chunk_chunk_size_4", expanded_signature)
        self.assertEqual(action_space_signature(restored), expanded_signature)

    def test_latent_patch_codec_is_deterministic_and_decodable(self):
        codec = LatentPatchActionCodec(patch_size=3, latent_size=4)
        first = codec.encode("one two three four")
        second = codec.encode("one two three four")

        self.assertEqual([action.payload for action in first], [action.payload for action in second])
        self.assertEqual(codec.decode(first), "one two three four")
        self.assertEqual(first[0].kind, "latent_patch")

    def test_command_codec_accepts_json_or_plain_text(self):
        codec = CommandActionCodec()
        command = codec.encode('{"name":"search","args":{"query":"rl"}}')
        fallback = codec.encode("say hello")

        self.assertEqual(command[0].payload["name"], "search")
        self.assertEqual(fallback[0].payload["name"], "say")

    def test_reasoning_step_codec_uses_lines_as_decisions(self):
        actions = ReasoningStepCodec().encode("Plan\nAct\nCheck")

        self.assertEqual(len(actions), 3)
        self.assertEqual(actions[1].payload, "Act")

    def test_action_logprob_stats_reads_typed_fields_and_metadata(self):
        actions = [
            ActionUnit(
                kind="chunk",
                payload=("alpha", "beta"),
                token_count=2,
                old_logprob=-2.0,
                new_logprob=-1.5,
                reference_logprob=-2.5,
            ),
            ActionUnit(
                kind="latent_patch",
                payload=(0.1, 0.2),
                token_count=2,
                metadata={
                    "logprob": -1.0,
                    "train/logprob": -0.7,
                    "ref/logprob": -1.2,
                },
            ),
            ActionUnit(kind="command", payload={"name": "noop"}, token_count=1),
        ]

        stats = action_logprob_stats(actions)

        self.assertEqual(stats.action_units, 3)
        self.assertAlmostEqual(stats.old_logprob_coverage, 2 / 3)
        self.assertAlmostEqual(stats.new_logprob_coverage, 2 / 3)
        self.assertAlmostEqual(stats.reference_logprob_coverage, 2 / 3)
        self.assertAlmostEqual(stats.old_new_logprob_delta_mean, 0.4)
        self.assertAlmostEqual(stats.old_new_logprob_abs_delta_mean, 0.4)
        self.assertAlmostEqual(stats.old_reference_logprob_delta_mean, 0.35)
        self.assertGreater(stats.importance_ratio_mean, 1.0)

    def test_adaptive_action_space_promotes_larger_chunks_from_objective_signal(self):
        action_space = AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4)

        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/policy_improvement_objective_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
            }
        )

        self.assertIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )
        self.assertEqual(action_space.metrics()["action_space/promotions"], 1.0)

    def test_adaptive_action_space_requires_observed_pulls_for_promotion(self):
        action_space = AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4)

        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/policy_improvement_objective_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
            }
        )

        self.assertNotIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )
        self.assertEqual(action_space.metrics()["action_space/promotions"], 0.0)

        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/policy_improvement_objective_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
            }
        )

        self.assertIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )
        self.assertEqual(action_space.metrics()["action_space/promotions"], 1.0)

    def test_adaptive_action_space_can_skip_promotions_for_stale_feedback(self):
        action_space = AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4)

        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 3.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
            },
            allow_promotions=False,
        )

        self.assertNotIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )
        self.assertEqual(action_space.metrics()["action_space/promotions"], 0.0)

    def test_adaptive_action_space_uses_scored_objective_for_promotion(self):
        action_space = AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4)

        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 3.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/policy_improvement_objective_ema": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/marginal_objective_ema": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
            }
        )

        self.assertNotIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )
        self.assertEqual(action_space.metrics()["action_space/promotions"], 0.0)

    def test_adaptive_action_space_requires_parent_margin_for_promotion(self):
        action_space = AdaptiveActionSpace(
            min_chunk_size=2,
            max_chunk_size=4,
            promotion_parent_margin=0.1,
        )

        action_space.update_from_metrics(
            {
                "scheduler/arm/task_token/pulls": 3.0,
                "scheduler/arm/task_token/objective_score": 2.0,
                "scheduler/arm/task_token/action_quality_ema": 1.0,
                "scheduler/arm/task_token/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 3.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 2.05,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
            }
        )

        self.assertNotIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )

        action_space.update_from_metrics(
            {
                "scheduler/arm/task_token/pulls": 4.0,
                "scheduler/arm/task_token/objective_score": 2.0,
                "scheduler/arm/task_token/action_quality_ema": 1.0,
                "scheduler/arm/task_token/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 4.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 2.2,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
            }
        )

        self.assertIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )
        self.assertEqual(action_space.metrics()["action_space/promotions"], 1.0)

    def test_adaptive_action_space_can_require_parent_throughput_margin_for_promotion(self):
        action_space = AdaptiveActionSpace(
            min_chunk_size=2,
            max_chunk_size=4,
            promotion_parent_source_token_throughput_margin=0.25,
        )

        action_space.update_from_metrics(
            {
                "scheduler/arm/task_token/pulls": 3.0,
                "scheduler/arm/task_token/objective_score": 1.0,
                "scheduler/arm/task_token/action_quality_ema": 1.0,
                "scheduler/arm/task_token/unsafe_rate": 0.0,
                "scheduler/arm/task_token/source_tokens_per_dollar_second": 10.0,
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 3.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/source_tokens_per_dollar_second": 10.1,
            }
        )

        self.assertNotIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )

        action_space.update_from_metrics(
            {
                "scheduler/arm/task_token/pulls": 4.0,
                "scheduler/arm/task_token/objective_score": 1.0,
                "scheduler/arm/task_token/action_quality_ema": 1.0,
                "scheduler/arm/task_token/unsafe_rate": 0.0,
                "scheduler/arm/task_token/source_tokens_per_dollar_second": 10.0,
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 4.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/source_tokens_per_dollar_second": 10.5,
            }
        )

        self.assertIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )
        self.assertEqual(action_space.metrics()["action_space/promotions"], 1.0)

    def test_adaptive_action_space_reports_promotion_decision_payoff(self):
        action_space = AdaptiveActionSpace(
            min_chunk_size=2,
            max_chunk_size=4,
            promotion_parent_margin=0.1,
        )
        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 3.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 2.2,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/source_tokens_per_dollar_second": 8.0,
            }
        )
        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 5.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 2.2,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/source_tokens_per_dollar_second": 8.0,
                "scheduler/arm/task_chunk_chunk_size_4/pulls": 2.0,
                "scheduler/arm/task_chunk_chunk_size_4/objective_score": 2.9,
                "scheduler/arm/task_chunk_chunk_size_4/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_4/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_4/semantic_bandwidth_tokens_per_decision": 4.0,
                "scheduler/arm/task_chunk_chunk_size_4/source_tokens_per_dollar_second": 9.5,
            }
        )

        metrics = action_space.metrics()
        prefix = (
            "action_space/decision/"
            "promotion_chunk_chunk_size_4_from_chunk_chunk_size_2"
        )
        self.assertEqual(metrics["action_space/decision/decisions"], 1.0)
        self.assertEqual(
            metrics["action_space/decision/post_decision_observations"],
            2.0,
        )
        self.assertAlmostEqual(
            metrics["action_space/decision/realized_objective_payoff"],
            1.4,
        )
        self.assertAlmostEqual(
            metrics[
                "action_space/decision/"
                "mean_realized_objective_payoff_per_decision"
            ],
            1.4,
        )
        self.assertAlmostEqual(
            metrics[
                "action_space/decision/"
                "mean_realized_objective_payoff_per_post_decision_observation"
            ],
            0.7,
        )
        self.assertEqual(metrics[f"{prefix}/decisions"], 1.0)
        self.assertEqual(metrics[f"{prefix}/promotion"], 1.0)
        self.assertEqual(metrics[f"{prefix}/target_pulls"], 2.0)
        self.assertEqual(metrics[f"{prefix}/parent_pulls"], 5.0)
        self.assertAlmostEqual(
            metrics[f"{prefix}/objective_delta_vs_parent"],
            0.7,
        )
        self.assertAlmostEqual(
            metrics[f"{prefix}/estimated_objective_payoff"],
            0.7,
        )
        self.assertEqual(metrics[f"{prefix}/decision_target_pulls"], 0.0)
        self.assertEqual(metrics[f"{prefix}/decision_parent_pulls"], 3.0)
        self.assertEqual(metrics[f"{prefix}/post_decision_target_pulls"], 2.0)
        self.assertEqual(metrics[f"{prefix}/post_decision_parent_pulls"], 2.0)
        self.assertEqual(metrics[f"{prefix}/post_decision_observations"], 2.0)
        self.assertAlmostEqual(
            metrics[f"{prefix}/realized_objective_payoff"],
            1.4,
        )
        self.assertAlmostEqual(
            metrics[f"{prefix}/mean_realized_objective_payoff_per_decision"],
            1.4,
        )
        self.assertAlmostEqual(
            metrics[
                f"{prefix}/"
                "mean_realized_objective_payoff_per_post_decision_observation"
            ],
            0.7,
        )
        self.assertAlmostEqual(
            metrics[f"{prefix}/source_token_throughput_delta_vs_parent"],
            1.5,
        )
        self.assertAlmostEqual(
            metrics[f"{prefix}/realized_source_token_throughput_payoff"],
            3.0,
        )
        self.assertAlmostEqual(
            metrics[
                "action_space/decision/realized_source_token_throughput_payoff"
            ],
            3.0,
        )
        self.assertAlmostEqual(
            metrics[
                "action_space/decision/"
                "mean_realized_source_token_throughput_payoff_per_decision"
            ],
            3.0,
        )
        self.assertAlmostEqual(
            metrics[
                "action_space/decision/"
                "mean_realized_source_token_throughput_payoff_per_"
                "post_decision_observation"
            ],
            1.5,
        )
        self.assertAlmostEqual(
            metrics[
                f"{prefix}/"
                "mean_realized_source_token_throughput_payoff_per_decision"
            ],
            3.0,
        )
        self.assertAlmostEqual(
            metrics[
                f"{prefix}/"
                "mean_realized_source_token_throughput_payoff_per_"
                "post_decision_observation"
            ],
            1.5,
        )

    def test_adaptive_action_space_can_promote_latent_patch_from_chunk_signal(self):
        action_space = AdaptiveActionSpace(
            min_chunk_size=2,
            max_chunk_size=4,
            promote_latent_patches=True,
            latent_patch_latent_size=3,
        )

        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/policy_improvement_objective_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
            }
        )

        codec_keys = [action_codec_key(codec) for codec in action_space.codecs]
        self.assertIn("chunk(chunk_size=4)", codec_keys)
        self.assertIn(
            "latent_patch(latent_size=3,patch_size=2)",
            codec_keys,
        )
        self.assertEqual(action_space.metrics()["action_space/promotions"], 2.0)

    def test_adaptive_action_space_can_promote_latent_patch_at_max_chunk_size(self):
        action_space = AdaptiveActionSpace(
            min_chunk_size=2,
            max_chunk_size=2,
            promote_latent_patches=True,
            latent_patch_latent_size=3,
        )

        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/policy_improvement_objective_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
            }
        )

        codec_keys = [action_codec_key(codec) for codec in action_space.codecs]
        self.assertEqual(
            codec_keys.count("chunk(chunk_size=2)"),
            1,
        )
        self.assertIn(
            "latent_patch(latent_size=3,patch_size=2)",
            codec_keys,
        )
        self.assertEqual(action_space.metrics()["action_space/promotions"], 1.0)

    def test_adaptive_action_space_requires_observed_semantic_bandwidth(self):
        action_space = AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4)

        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/policy_improvement_objective_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
            }
        )

        self.assertNotIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )
        self.assertEqual(action_space.metrics()["action_space/promotions"], 0.0)

    def test_adaptive_action_space_can_require_logprob_coverage_for_promotion(self):
        action_space = AdaptiveActionSpace(
            min_chunk_size=2,
            max_chunk_size=4,
            promotion_min_old_logprob_coverage=1.0,
            promotion_min_new_logprob_coverage=1.0,
        )
        base_metrics = {
            "scheduler/arm/task_chunk_chunk_size_2/pulls": 1.0,
            "scheduler/arm/task_chunk_chunk_size_2/policy_improvement_objective_ema": 1.0,
            "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
            "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
            "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
        }

        action_space.update_from_metrics(base_metrics)

        self.assertNotIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )

        action_space.update_from_metrics(
            {
                **base_metrics,
                "scheduler/arm/task_chunk_chunk_size_2/old_logprob_coverage": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/new_logprob_coverage": 1.0,
            }
        )

        self.assertIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )
        self.assertEqual(action_space.metrics()["action_space/promotions"], 1.0)

    def test_adaptive_action_space_does_not_promote_unsafe_chunks(self):
        action_space = AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4)

        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/policy_improvement_objective_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
            }
        )

        self.assertNotIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )
        self.assertEqual(action_space.metrics()["action_space/promotions"], 0.0)

    def test_adaptive_action_space_does_not_promote_failed_chunks(self):
        action_space = AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4)

        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/policy_improvement_objective_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/failure_rate": 0.5,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
            }
        )

        self.assertNotIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )
        self.assertEqual(action_space.metrics()["action_space/promotions"], 0.0)

    def test_adaptive_action_space_does_not_promote_drifty_chunks(self):
        action_space = AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4)

        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/policy_improvement_objective_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/reconstruction_max_drift": 0.2,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
            }
        )

        self.assertNotIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )
        self.assertEqual(action_space.metrics()["action_space/promotions"], 0.0)

    def test_adaptive_action_space_demotes_bad_promoted_chunks(self):
        action_space = AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=8)
        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 3.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
            }
        )
        self.assertIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )

        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 4.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
                "scheduler/arm/task_chunk_chunk_size_4/pulls": 2.0,
                "scheduler/arm/task_chunk_chunk_size_4/objective_score": 0.0,
                "scheduler/arm/task_chunk_chunk_size_4/action_quality_ema": 0.0,
                "scheduler/arm/task_chunk_chunk_size_4/unsafe_rate": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 4.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
            }
        )
        metrics = action_space.metrics()

        self.assertNotIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )
        self.assertEqual(metrics["action_space/demotions"], 1.0)
        self.assertEqual(metrics["action_space/disabled_codecs"], 1.0)
        self.assertEqual(
            metrics["action_space/codec/chunk_chunk_size_4/disabled"],
            1.0,
        )
        self.assertEqual(metrics["action_space/max_chunk_size"], 2.0)

    def test_adaptive_action_space_uses_scored_objective_for_demotion(self):
        action_space = AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=8)
        action_space.add_codec(ChunkActionCodec(chunk_size=4))

        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_4/pulls": 2.0,
                "scheduler/arm/task_chunk_chunk_size_4/objective_score": 0.0,
                "scheduler/arm/task_chunk_chunk_size_4/policy_improvement_objective_ema": 2.0,
                "scheduler/arm/task_chunk_chunk_size_4/marginal_objective_ema": 2.0,
                "scheduler/arm/task_chunk_chunk_size_4/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_4/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_4/semantic_bandwidth_tokens_per_decision": 4.0,
            }
        )

        self.assertNotIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )
        self.assertEqual(action_space.metrics()["action_space/demotions"], 1.0)

    def test_adaptive_action_space_can_skip_demotions_for_rollout_feedback(self):
        action_space = AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=8)
        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 3.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
            }
        )

        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_4/pulls": 2.0,
                "scheduler/arm/task_chunk_chunk_size_4/objective_score": 0.0,
                "scheduler/arm/task_chunk_chunk_size_4/action_quality_ema": 0.0,
                "scheduler/arm/task_chunk_chunk_size_4/unsafe_rate": 1.0,
            },
            allow_demotions=False,
        )

        self.assertIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )
        self.assertEqual(action_space.metrics()["action_space/demotions"], 0.0)

    def test_adaptive_action_space_demotes_drifty_promoted_chunks(self):
        action_space = AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=8)
        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 3.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
            }
        )
        self.assertIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )

        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_4/pulls": 2.0,
                "scheduler/arm/task_chunk_chunk_size_4/objective_score": 2.0,
                "scheduler/arm/task_chunk_chunk_size_4/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_4/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_4/reconstruction_max_drift": 0.2,
                "scheduler/arm/task_chunk_chunk_size_4/semantic_bandwidth_tokens_per_decision": 4.0,
            }
        )

        self.assertNotIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )
        self.assertEqual(action_space.metrics()["action_space/demotions"], 1.0)

    def test_adaptive_action_space_demotes_failed_promoted_chunks(self):
        action_space = AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=8)
        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 3.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
            }
        )

        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_4/pulls": 2.0,
                "scheduler/arm/task_chunk_chunk_size_4/objective_score": 2.0,
                "scheduler/arm/task_chunk_chunk_size_4/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_4/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_4/failure_rate": 0.5,
            }
        )

        self.assertNotIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )
        self.assertEqual(action_space.metrics()["action_space/demotions"], 1.0)

    def test_adaptive_action_space_demotes_chunk_when_parent_has_better_objective(self):
        action_space = AdaptiveActionSpace(
            min_chunk_size=2,
            max_chunk_size=8,
            demotion_parent_margin=0.25,
        )
        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 4.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
            }
        )
        self.assertIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )

        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 5.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 1.8,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
                "scheduler/arm/task_chunk_chunk_size_4/pulls": 2.0,
                "scheduler/arm/task_chunk_chunk_size_4/objective_score": 1.4,
                "scheduler/arm/task_chunk_chunk_size_4/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_4/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_4/semantic_bandwidth_tokens_per_decision": 4.0,
            }
        )

        self.assertNotIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )
        self.assertEqual(action_space.metrics()["action_space/demotions"], 1.0)

    def test_adaptive_action_space_demotes_chunk_when_parent_has_better_throughput(self):
        action_space = AdaptiveActionSpace(
            min_chunk_size=2,
            max_chunk_size=4,
            demotion_parent_source_token_throughput_margin=0.5,
        )
        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/source_tokens_per_dollar_second": 12.0,
            }
        )
        self.assertIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )

        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/source_tokens_per_dollar_second": 12.0,
                "scheduler/arm/task_chunk_chunk_size_4/pulls": 2.0,
                "scheduler/arm/task_chunk_chunk_size_4/objective_score": 1.2,
                "scheduler/arm/task_chunk_chunk_size_4/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_4/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_4/semantic_bandwidth_tokens_per_decision": 4.0,
                "scheduler/arm/task_chunk_chunk_size_4/source_tokens_per_dollar_second": 11.0,
            }
        )

        self.assertNotIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )
        self.assertEqual(action_space.metrics()["action_space/demotions"], 1.0)

    def test_adaptive_action_space_reports_demotion_decision_payoff(self):
        action_space = AdaptiveActionSpace(
            min_chunk_size=2,
            max_chunk_size=4,
            demotion_parent_margin=0.1,
        )
        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 3.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/source_tokens_per_dollar_second": 8.0,
            }
        )
        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 4.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/source_tokens_per_dollar_second": 8.0,
                "scheduler/arm/task_chunk_chunk_size_4/pulls": 2.0,
                "scheduler/arm/task_chunk_chunk_size_4/objective_score": 0.9,
                "scheduler/arm/task_chunk_chunk_size_4/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_4/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_4/semantic_bandwidth_tokens_per_decision": 4.0,
                "scheduler/arm/task_chunk_chunk_size_4/source_tokens_per_dollar_second": 6.5,
            }
        )

        metrics = action_space.metrics()
        prefix = (
            "action_space/decision/"
            "demotion_chunk_chunk_size_4_from_chunk_chunk_size_2"
        )
        self.assertEqual(metrics[f"{prefix}/decisions"], 1.0)
        self.assertEqual(metrics[f"{prefix}/demotion"], 1.0)
        self.assertEqual(metrics[f"{prefix}/target_pulls"], 2.0)
        self.assertEqual(metrics[f"{prefix}/parent_pulls"], 4.0)
        self.assertAlmostEqual(
            metrics[f"{prefix}/objective_delta_vs_parent"],
            -1.1,
        )
        self.assertAlmostEqual(
            metrics[f"{prefix}/estimated_objective_payoff"],
            1.1,
        )
        self.assertAlmostEqual(
            metrics[f"{prefix}/source_token_throughput_delta_vs_parent"],
            -1.5,
        )

    def test_adaptive_action_space_reports_realized_demotion_payoff(self):
        action_space = AdaptiveActionSpace(
            min_chunk_size=2,
            max_chunk_size=4,
            demotion_parent_margin=0.1,
        )
        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 3.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/source_tokens_per_dollar_second": 8.0,
            }
        )
        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 4.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/source_tokens_per_dollar_second": 8.0,
                "scheduler/arm/task_chunk_chunk_size_4/pulls": 2.0,
                "scheduler/arm/task_chunk_chunk_size_4/objective_score": 0.9,
                "scheduler/arm/task_chunk_chunk_size_4/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_4/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_4/semantic_bandwidth_tokens_per_decision": 4.0,
                "scheduler/arm/task_chunk_chunk_size_4/source_tokens_per_dollar_second": 6.5,
            }
        )
        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 6.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 2.1,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/source_tokens_per_dollar_second": 8.5,
                "scheduler/arm/task_chunk_chunk_size_4/pulls": 2.0,
                "scheduler/arm/task_chunk_chunk_size_4/objective_score": 0.9,
                "scheduler/arm/task_chunk_chunk_size_4/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_4/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_4/semantic_bandwidth_tokens_per_decision": 4.0,
                "scheduler/arm/task_chunk_chunk_size_4/source_tokens_per_dollar_second": 6.5,
            }
        )

        metrics = action_space.metrics()
        prefix = (
            "action_space/decision/"
            "demotion_chunk_chunk_size_4_from_chunk_chunk_size_2"
        )
        self.assertEqual(metrics[f"{prefix}/decision_target_pulls"], 2.0)
        self.assertEqual(metrics[f"{prefix}/decision_parent_pulls"], 4.0)
        self.assertEqual(metrics[f"{prefix}/post_decision_target_pulls"], 0.0)
        self.assertEqual(metrics[f"{prefix}/post_decision_parent_pulls"], 2.0)
        self.assertEqual(metrics[f"{prefix}/post_decision_observations"], 2.0)
        self.assertAlmostEqual(
            metrics[f"{prefix}/estimated_objective_payoff"],
            1.2,
        )
        self.assertAlmostEqual(
            metrics[f"{prefix}/realized_objective_payoff"],
            2.4,
        )
        self.assertAlmostEqual(
            metrics[f"{prefix}/realized_source_token_throughput_payoff"],
            4.0,
        )

    def test_adaptive_action_space_demotes_negative_promotion_decision_payoff(self):
        action_space = AdaptiveActionSpace(
            min_chunk_size=2,
            max_chunk_size=4,
            demotion_objective_threshold=-100.0,
            demotion_parent_margin=10.0,
            demotion_decision_min_observations=2,
        )
        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 3.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
            }
        )
        self.assertIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )

        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 4.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
                "scheduler/arm/task_chunk_chunk_size_4/pulls": 1.0,
                "scheduler/arm/task_chunk_chunk_size_4/objective_score": 1.0,
                "scheduler/arm/task_chunk_chunk_size_4/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_4/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_4/semantic_bandwidth_tokens_per_decision": 4.0,
            }
        )
        self.assertIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )

        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 5.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
                "scheduler/arm/task_chunk_chunk_size_4/pulls": 2.0,
                "scheduler/arm/task_chunk_chunk_size_4/objective_score": 1.0,
                "scheduler/arm/task_chunk_chunk_size_4/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_4/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_4/semantic_bandwidth_tokens_per_decision": 4.0,
            }
        )

        metrics = action_space.metrics()
        prefix = (
            "action_space/decision/"
            "promotion_chunk_chunk_size_4_from_chunk_chunk_size_2"
        )

        self.assertNotIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )
        self.assertEqual(metrics["action_space/demotions"], 1.0)
        self.assertEqual(metrics["action_space/decision_payoff_demotions"], 1.0)
        self.assertEqual(metrics["action_space/codec/chunk_chunk_size_4/disabled"], 1.0)
        self.assertEqual(
            metrics["action_space/demotion_decision_min_observations"],
            2.0,
        )
        self.assertAlmostEqual(
            metrics[f"{prefix}/realized_objective_payoff"],
            -2.0,
        )

    def test_adaptive_action_space_source_throughput_payoff_demotion_is_opt_in(self):
        action_space = AdaptiveActionSpace(
            min_chunk_size=2,
            max_chunk_size=4,
            demotion_objective_threshold=-100.0,
            demotion_parent_margin=10.0,
            demotion_decision_payoff_threshold=-100.0,
            demotion_decision_min_observations=2,
        )
        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 3.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/source_tokens_per_dollar_second": 8.0,
            }
        )
        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 5.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/source_tokens_per_dollar_second": 8.0,
                "scheduler/arm/task_chunk_chunk_size_4/pulls": 2.0,
                "scheduler/arm/task_chunk_chunk_size_4/objective_score": 2.1,
                "scheduler/arm/task_chunk_chunk_size_4/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_4/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_4/semantic_bandwidth_tokens_per_decision": 4.0,
                "scheduler/arm/task_chunk_chunk_size_4/source_tokens_per_dollar_second": 6.0,
            }
        )

        metrics = action_space.metrics()

        self.assertIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )
        self.assertEqual(metrics["action_space/demotions"], 0.0)
        self.assertEqual(
            metrics[
                "action_space/"
                "demotion_decision_source_token_throughput_payoff_enabled"
            ],
            0.0,
        )
        self.assertEqual(
            metrics["action_space/source_token_throughput_payoff_demotions"],
            0.0,
        )

    def test_adaptive_action_space_demotes_negative_source_throughput_payoff(self):
        action_space = AdaptiveActionSpace(
            min_chunk_size=2,
            max_chunk_size=4,
            demotion_objective_threshold=-100.0,
            demotion_parent_margin=10.0,
            demotion_decision_payoff_threshold=-100.0,
            demotion_decision_source_token_throughput_payoff_threshold=0.0,
            demotion_decision_min_observations=2,
        )
        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 3.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/source_tokens_per_dollar_second": 8.0,
            }
        )
        self.assertIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )

        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 4.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/source_tokens_per_dollar_second": 8.0,
                "scheduler/arm/task_chunk_chunk_size_4/pulls": 1.0,
                "scheduler/arm/task_chunk_chunk_size_4/objective_score": 2.1,
                "scheduler/arm/task_chunk_chunk_size_4/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_4/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_4/semantic_bandwidth_tokens_per_decision": 4.0,
                "scheduler/arm/task_chunk_chunk_size_4/source_tokens_per_dollar_second": 6.0,
            }
        )
        self.assertIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )

        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 5.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/source_tokens_per_dollar_second": 8.0,
                "scheduler/arm/task_chunk_chunk_size_4/pulls": 2.0,
                "scheduler/arm/task_chunk_chunk_size_4/objective_score": 2.1,
                "scheduler/arm/task_chunk_chunk_size_4/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_4/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_4/semantic_bandwidth_tokens_per_decision": 4.0,
                "scheduler/arm/task_chunk_chunk_size_4/source_tokens_per_dollar_second": 6.0,
            }
        )

        metrics = action_space.metrics()
        prefix = (
            "action_space/decision/"
            "promotion_chunk_chunk_size_4_from_chunk_chunk_size_2"
        )

        self.assertNotIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )
        self.assertEqual(metrics["action_space/demotions"], 1.0)
        self.assertEqual(metrics["action_space/decision_payoff_demotions"], 1.0)
        self.assertEqual(
            metrics["action_space/source_token_throughput_payoff_demotions"],
            1.0,
        )
        self.assertEqual(
            metrics[
                "action_space/"
                "demotion_decision_source_token_throughput_payoff_enabled"
            ],
            1.0,
        )
        self.assertEqual(
            metrics[
                "action_space/"
                "demotion_decision_source_token_throughput_payoff_threshold"
            ],
            0.0,
        )
        self.assertAlmostEqual(
            metrics[f"{prefix}/realized_objective_payoff"],
            0.2,
        )
        self.assertAlmostEqual(
            metrics[f"{prefix}/realized_source_token_throughput_payoff"],
            -4.0,
        )

    def test_adaptive_action_space_demotes_bad_latent_patch_candidate(self):
        action_space = AdaptiveActionSpace(
            min_chunk_size=2,
            max_chunk_size=4,
            promote_latent_patches=True,
            latent_patch_latent_size=3,
        )
        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 3.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
            }
        )
        self.assertIn(
            "latent_patch(latent_size=3,patch_size=2)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )

        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 4.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
                "scheduler/arm/task_latent_patch_latent_size_3_patch_size_2/pulls": 2.0,
                "scheduler/arm/task_latent_patch_latent_size_3_patch_size_2/objective_score": 0.0,
                "scheduler/arm/task_latent_patch_latent_size_3_patch_size_2/action_quality_ema": 0.0,
                "scheduler/arm/task_latent_patch_latent_size_3_patch_size_2/unsafe_rate": 1.0,
                "scheduler/arm/task_latent_patch_latent_size_3_patch_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
            }
        )

        self.assertNotIn(
            "latent_patch(latent_size=3,patch_size=2)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )
        metrics = action_space.metrics()
        self.assertEqual(metrics["action_space/demotions"], 1.0)
        self.assertEqual(
            metrics[
                "action_space/codec/latent_patch_latent_size_3_patch_size_2/disabled"
            ],
            1.0,
        )

    def test_adaptive_action_space_demotes_dependent_latent_patch_with_chunk(self):
        action_space = AdaptiveActionSpace(
            min_chunk_size=2,
            max_chunk_size=4,
            promote_latent_patches=True,
            latent_patch_latent_size=3,
        )
        action_space.add_codec(ChunkActionCodec(chunk_size=4))
        action_space.add_codec(
            LatentPatchActionCodec(patch_size=4, latent_size=3)
        )

        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_4/pulls": 2.0,
                "scheduler/arm/task_chunk_chunk_size_4/objective_score": 0.0,
                "scheduler/arm/task_chunk_chunk_size_4/action_quality_ema": 0.0,
                "scheduler/arm/task_chunk_chunk_size_4/unsafe_rate": 1.0,
                "scheduler/arm/task_latent_patch_latent_size_3_patch_size_4/pulls": 0.0,
            }
        )

        codec_keys = [action_codec_key(codec) for codec in action_space.codecs]
        metrics = action_space.metrics()

        self.assertNotIn("chunk(chunk_size=4)", codec_keys)
        self.assertNotIn("latent_patch(latent_size=3,patch_size=4)", codec_keys)
        self.assertEqual(metrics["action_space/demotions"], 2.0)
        self.assertEqual(
            metrics["action_space/codec/chunk_chunk_size_4/disabled"],
            1.0,
        )
        self.assertEqual(
            metrics[
                "action_space/codec/latent_patch_latent_size_3_patch_size_4/disabled"
            ],
            1.0,
        )

    def test_adaptive_action_space_does_not_repromote_demoted_chunks(self):
        action_space = AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4)
        action_space.add_codec(ChunkActionCodec(chunk_size=4))
        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_4/pulls": 2.0,
                "scheduler/arm/task_chunk_chunk_size_4/objective_score": 0.0,
                "scheduler/arm/task_chunk_chunk_size_4/action_quality_ema": 0.0,
                "scheduler/arm/task_chunk_chunk_size_4/unsafe_rate": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 4.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
            }
        )
        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 5.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
            }
        )

        self.assertNotIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )
        self.assertEqual(action_space.metrics()["action_space/promotions"], 0.0)

    def test_adaptive_action_space_keeps_min_chunk_as_baseline(self):
        action_space = AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4)

        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 3.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 1.0,
            }
        )

        self.assertIn(
            "chunk(chunk_size=2)",
            [action_codec_key(codec) for codec in action_space.codecs],
        )
        self.assertEqual(action_space.metrics()["action_space/demotions"], 0.0)

    def test_adaptive_action_space_state_round_trips_promotions_and_demotions(self):
        action_space = AdaptiveActionSpace(
            min_chunk_size=2,
            max_chunk_size=8,
            promotion_parent_margin=0.1,
            demotion_parent_margin=0.25,
            promotion_parent_source_token_throughput_margin=0.5,
            demotion_parent_source_token_throughput_margin=0.75,
            promote_latent_patches=True,
            latent_patch_latent_size=3,
            promotion_min_pulls=2,
            promotion_max_reconstruction_drift=0.03,
            demotion_max_reconstruction_drift=0.08,
            demotion_decision_payoff_threshold=-0.5,
            demotion_decision_source_token_throughput_payoff_threshold=-1.5,
            demotion_decision_min_observations=3,
            demote_on_stale_feedback=True,
        )
        action_space.update_from_metrics(
            {
                "scheduler/arm/task_token/pulls": 3.0,
                "scheduler/arm/task_token/objective_score": 0.0,
                "scheduler/arm/task_token/action_quality_ema": 1.0,
                "scheduler/arm/task_token/unsafe_rate": 0.0,
                "scheduler/arm/task_token/source_tokens_per_dollar_second": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 3.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/semantic_bandwidth_tokens_per_decision": 2.0,
                "scheduler/arm/task_chunk_chunk_size_2/source_tokens_per_dollar_second": 2.0,
            }
        )
        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_4/pulls": 2.0,
                "scheduler/arm/task_chunk_chunk_size_4/objective_score": 0.0,
                "scheduler/arm/task_chunk_chunk_size_4/action_quality_ema": 0.0,
                "scheduler/arm/task_chunk_chunk_size_4/unsafe_rate": 1.0,
            }
        )

        state = action_space.state_dict()
        self.assertIn("decision_stats", state)
        restored = AdaptiveActionSpace(min_chunk_size=1, max_chunk_size=1)
        restored.load_state_dict(state)
        metrics = restored.metrics()

        self.assertEqual(restored.min_chunk_size, 2)
        self.assertEqual(restored.max_chunk_size, 8)
        self.assertEqual(restored.promotion_parent_margin, 0.1)
        self.assertEqual(restored.promotion_semantic_bandwidth_threshold, 1.0)
        self.assertEqual(
            restored.promotion_parent_source_token_throughput_margin,
            0.5,
        )
        self.assertEqual(
            metrics["action_space/promotion_parent_source_token_throughput_margin"],
            0.5,
        )
        self.assertEqual(restored.promotion_max_reconstruction_drift, 0.03)
        self.assertEqual(restored.promotion_min_pulls, 2)
        self.assertEqual(restored.demotion_parent_margin, 0.25)
        self.assertEqual(restored.demotion_semantic_bandwidth_threshold, 1.0)
        self.assertEqual(
            restored.demotion_parent_source_token_throughput_margin,
            0.75,
        )
        self.assertEqual(
            metrics["action_space/demotion_parent_source_token_throughput_margin"],
            0.75,
        )
        self.assertEqual(restored.demotion_max_reconstruction_drift, 0.08)
        self.assertEqual(restored.demotion_decision_payoff_threshold, -0.5)
        self.assertEqual(
            restored.demotion_decision_source_token_throughput_payoff_threshold,
            -1.5,
        )
        self.assertEqual(restored.demotion_decision_min_observations, 3)
        self.assertEqual(
            metrics["action_space/demotion_decision_payoff_threshold"],
            -0.5,
        )
        self.assertEqual(
            metrics[
                "action_space/"
                "demotion_decision_source_token_throughput_payoff_enabled"
            ],
            1.0,
        )
        self.assertEqual(
            metrics[
                "action_space/"
                "demotion_decision_source_token_throughput_payoff_threshold"
            ],
            -1.5,
        )
        self.assertEqual(
            metrics["action_space/demotion_decision_min_observations"],
            3.0,
        )
        self.assertTrue(restored.demote_on_stale_feedback)
        self.assertEqual(metrics["action_space/demote_on_stale_feedback"], 1.0)
        self.assertTrue(restored.promote_latent_patches)
        self.assertEqual(restored.latent_patch_latent_size, 3)
        self.assertEqual(metrics["action_space/promotions"], 2.0)
        self.assertEqual(metrics["action_space/demotions"], 1.0)
        self.assertEqual(
            metrics["action_space/source_token_throughput_payoff_demotions"],
            0.0,
        )
        self.assertEqual(metrics["action_space/disabled_codecs"], 1.0)
        self.assertIn(
            "chunk(chunk_size=2)",
            [action_codec_key(codec) for codec in restored.codecs],
        )
        self.assertNotIn(
            "chunk(chunk_size=4)",
            [action_codec_key(codec) for codec in restored.codecs],
        )
        self.assertEqual(
            metrics["action_space/codec/chunk_chunk_size_4/disabled"],
            1.0,
        )
        promotion_prefix = (
            "action_space/decision/"
            "promotion_chunk_chunk_size_4_from_chunk_chunk_size_2"
        )
        demotion_prefix = (
            "action_space/decision/"
            "demotion_chunk_chunk_size_4_from_chunk_chunk_size_2"
        )
        self.assertEqual(metrics[f"{promotion_prefix}/decisions"], 1.0)
        self.assertEqual(metrics[f"{demotion_prefix}/decisions"], 1.0)
        self.assertEqual(
            metrics[f"{demotion_prefix}/demotion"],
            1.0,
        )
        self.assertIn(
            "realized_objective_payoff",
            state["decision_stats"][
                "demotion_chunk_chunk_size_4_from_chunk_chunk_size_2"
            ],
        )
        self.assertIn(
            f"{demotion_prefix}/realized_objective_payoff",
            metrics,
        )

    def test_action_space_checkpoint_metadata_uses_named_state_key(self):
        action_space = AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4)

        metadata = action_space_checkpoint_metadata(action_space)

        self.assertEqual(sorted(metadata.keys()), [ACTION_SPACE_STATE_KEY])
        self.assertEqual(metadata[ACTION_SPACE_STATE_KEY]["version"], 1)


if __name__ == "__main__":
    unittest.main()
