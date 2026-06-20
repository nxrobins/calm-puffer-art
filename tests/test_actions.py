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
        action_space = AdaptiveActionSpace(min_chunk_size=2, max_chunk_size=8)
        action_space.update_from_metrics(
            {
                "scheduler/arm/task_chunk_chunk_size_2/pulls": 3.0,
                "scheduler/arm/task_chunk_chunk_size_2/objective_score": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/action_quality_ema": 1.0,
                "scheduler/arm/task_chunk_chunk_size_2/unsafe_rate": 0.0,
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
        self.assertEqual(metrics["action_space/promotions"], 1.0)
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
