from LoadedLookupTable import LoadedLookupTable
import unittest

class LoadedLookupTableTest(unittest.TestCase):
  
  @classmethod
  def setUpClass(cls):
    cls.lookup_table = LoadedLookupTable.load()

  def test_known_indices(self):
    self.assertEqual(1738, LoadedLookupTable.PADDING)
    self.assertEqual(1739, LoadedLookupTable.UNKNOWN)
    self.assertEqual(LoadedLookupTable.PADDING, self.lookup_table.query_index('PADDING'))
    self.assertEqual(LoadedLookupTable.UNKNOWN, self.lookup_table.query_index('UNKNOWN'))

if '__main__' == __name__:
  unittest.main()
