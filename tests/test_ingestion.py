import unittest
from rag.ingest import normalize_text, parse_paragraphs, chunk_paragraphs

class IngestionTests(unittest.TestCase):
    def test_nfc_normalization(self):
        self.assertEqual(normalize_text('कि\u200d'), normalize_text('कि'))

    def test_section_heading_is_preserved(self):
        paras = parse_paragraphs('8 · इसरो के वर्ष और SLV-III\nजुलाई 1980 में SLV-III ने रोहिणी उपग्रह को कक्षा में स्थापित किया।')
        self.assertTrue(paras)
        self.assertIn('इसरो के वर्ष और SLV-III', paras[0][0])

    def test_practice_questions_are_tagged(self):
        paras = parse_paragraphs('19 · विचार एवं बोध-प्रश्न\n1. कलाम को भारत रत्न किस वर्ष मिला?')
        chunks = chunk_paragraphs(paras, 21)
        self.assertTrue(chunks)
        self.assertTrue(all(x['source_type'] == 'practice_question' for x in chunks))

    def test_chunk_metadata(self):
        chunks = chunk_paragraphs([('पुरस्कार एवं सम्मान', 'कलाम को भारत रत्न 1997 में प्रदान किया गया।')], 16)
        self.assertEqual(chunks[0]['page'], 16)
        self.assertEqual(chunks[0]['section'], 'पुरस्कार एवं सम्मान')
        self.assertTrue(chunks[0]['chunk_id'])

if __name__ == '__main__':
    unittest.main()
