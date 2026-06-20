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
    action_space_checkpoint_metadata,
    semantic_bandwidth,
)


class ActionCodecTests(unittest.TestCase):
    def test_chunk_codec_round_trips_with_higher_bandwidth_than_tokens(self):
        text = "alpha beta gamma delta epsilon"
        token_actions = TokenActionCodec().encode(text)
        chunk_actions = ChunkActionCodec(chunk_size=2).encode(text)

        self.assertEqual(ChunkActionCodec(chunk_size=2).decode(chunk_actions), text)
        self.assertEqual(len(token_actions), 5)
        self.assertEqual(len(chunk_actions), 3)
        self.assertGreater(semantic_bandwidth(chunk_actions), semantic_bandwidth(token_actions))

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

    def test_adaptive_action_space_promotes_larger_chunks_from_objective_signal(self):
        action_space = AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4)

        action_space.update_from_metrics(
            {
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

    def test_adaptive_action_space_can_promote_latent_patch_from_chunk_signal(self):
        action_space = AdaptiveActionSpace(
            min_chunk_size=2,
            max_chunk_size=4,
            promote_latent_patches=True,
            latent_patch_latent_size=3,
        )

        action_space.update_from_metrics(
            {
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

    def test_adaptive_action_space_does_not_promote_unsafe_chunks(self):
        action_space = AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4)

        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/policy_improvement_objective_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 1.0,
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
                "scheduler/arm/task_chunk_chunk_size_2/policy_improvement_objective_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
                "scheduler/arm/task_chunk_chunk_size_2/failure_rate": 0.5,
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
        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_4/pulls": 2.0,
                "scheduler/arm/task_chunk_chunk_size_4/objective_score": 0.0,
                "scheduler/arm/task_chunk_chunk_size_4/action_quality_ema": 0.0,
                "scheduler/arm/task_chunk_chunk_size_4/unsafe_rate": 1.0,
            }
        )

        restored = AdaptiveActionSpace(min_chunk_size=1, max_chunk_size=1)
        restored.load_state_dict(action_space.state_dict())
        metrics = restored.metrics()

        self.assertEqual(restored.min_chunk_size, 2)
        self.assertEqual(restored.max_chunk_size, 8)
        self.assertEqual(restored.promotion_parent_margin, 0.1)
        self.assertEqual(restored.promotion_semantic_bandwidth_threshold, 1.0)
        self.assertEqual(restored.demotion_parent_margin, 0.25)
        self.assertEqual(restored.demotion_semantic_bandwidth_threshold, 1.0)
        self.assertTrue(restored.promote_latent_patches)
        self.assertEqual(restored.latent_patch_latent_size, 3)
        self.assertEqual(metrics["action_space/promotions"], 2.0)
        self.assertEqual(metrics["action_space/demotions"], 1.0)
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

    def test_action_space_checkpoint_metadata_uses_named_state_key(self):
        action_space = AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=4)

        metadata = action_space_checkpoint_metadata(action_space)

        self.assertEqual(sorted(metadata.keys()), [ACTION_SPACE_STATE_KEY])
        self.assertEqual(metadata[ACTION_SPACE_STATE_KEY]["version"], 1)


if __name__ == "__main__":
    unittest.main()
