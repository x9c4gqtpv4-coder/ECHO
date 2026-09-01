import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from batch_color.safety import file_hash
from scripts.migrate_flat_sku_reports import migrate


class ReportMigrationTests(unittest.TestCase):
    def test_migration_repairs_paths_backs_up_json_and_never_edits_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "校色输出"
            sku_root = root / "sz123"
            output = sku_root / "校色成品" / "成品动作1.png"
            mask = sku_root / "蒙版" / "成品动作1" / "person.png"
            mask.parent.mkdir(parents=True)
            output.parent.mkdir(parents=True)
            Image.new("RGB", (8, 8), "red").save(output)
            Image.new("L", (8, 8), 255).save(mask)
            old_output = sku_root / "old-run" / "校色成品" / output.name
            old_mask = sku_root / ".old.processing" / "蒙版" / "成品动作1" / mask.name
            item = {
                "output": str(old_output),
                "output_sha256": file_hash(output),
                "mask_paths": {"person": str(old_mask)},
            }
            report = sku_root / "报告" / "item.json"
            report.parent.mkdir()
            report.write_text(json.dumps(item), encoding="utf-8")
            summary = {"items": [item], "configuration": {}}
            (sku_root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            before = file_hash(output)

            dry = migrate(root, apply=False)
            self.assertEqual(dry["reports_changed"], 2)
            self.assertTrue(Path(json.loads(report.read_text())["output"]).is_absolute())

            result = migrate(root, apply=True)
            self.assertEqual(result["reports_changed"], 2)
            self.assertTrue(Path(result["backup"]).is_file())
            repaired = json.loads(report.read_text())
            self.assertEqual(repaired["output"], "校色成品/成品动作1.png")
            self.assertEqual(repaired["mask_paths"]["person"], "蒙版/成品动作1/person.png")
            self.assertEqual(file_hash(output), before)
            self.assertFalse(result["images_modified"])


if __name__ == "__main__":
    unittest.main()
