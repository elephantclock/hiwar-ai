import unittest

from server.sentence_assembler import SentenceAssembler


class SentenceAssemblerTest(unittest.TestCase):
    def test_emits_complete_sentence_and_keeps_tail(self) -> None:
        assembler = SentenceAssembler()
        self.assertEqual(assembler.push("Your order is ready. It will"), ["Your order is ready."])
        self.assertEqual(assembler.push(" arrive today!"), ["It will arrive today!"])
        self.assertEqual(assembler.flush(), [])

    def test_splits_long_text_on_word_boundary(self) -> None:
        assembler = SentenceAssembler(max_chars=24)
        ready = assembler.push("This response should split before it gets too long")
        chunks = ready + assembler.flush()
        self.assertEqual(
            " ".join(chunks),
            "This response should split before it gets too long",
        )
        self.assertTrue(all(len(chunk) <= 24 for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
