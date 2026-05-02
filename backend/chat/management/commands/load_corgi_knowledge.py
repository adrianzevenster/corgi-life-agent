import json
from django.core.management.base import BaseCommand, CommandError

from chat.services.rag_qdrant import upsert_chunks

class Command(BaseCommand):
    help = "Load corgi breed facts/context into Qdrant from a JSONL file."

    def add_arguments(self, parser):
        parser.add_argument("path", type=str, help="Path to JSONL file with corgi facts")

    def handle(self, *args, **options):
        path = options["path"]

        chunks = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    chunks.append(
                        {
                            "id": obj["id"],
                            "text": obj["text"],
                            "source": obj.get("source", "curated"),
                            "tag": obj.get("tag", "corgi"),
                        }
                    )
        except FileNotFoundError:
            raise CommandError(f"File not found: {path}")
        except Exception as e:
            raise CommandError(f"Failed reading {path}: {e}")

        if not chunks:
            self.stdout.write(self.style.WARNING("No chunks found; nothing loaded."))
            return

        upsert_chunks(chunks)
        self.stdout.write(self.style.SUCCESS(f"Loaded/updated {len(chunks)} corgi knowledge chunks into Qdrant."))
