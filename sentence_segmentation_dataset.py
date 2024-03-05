import datasets


logger = datasets.logging.get_logger(__name__)


class SentenceSegmentationConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        """BuilderConfig for SentenceSegmentation
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SentenceSegmentationConfig, self).__init__(**kwargs)


class SentenceSegmentation(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        SentenceSegmentationConfig(name="sentence_segmentation", version=datasets.Version("1.0.0"), description="Sentence segmentation dataset")
    ]

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-S",
                                "B-E",
                            ],
                        ),
                    ),
                },
            ),
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        data_files = dl_manager.download_and_extract(self.config.data_files)
        if isinstance(data_files, (str, list, tuple)):
            files = data_files
            if isinstance(files, str):
                files = [files]
            return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"files": files})]
        splits = []
        for split_name, files in data_files.items():
            if isinstance(files, str):
                files = [files]
            splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files}))
        return splits

    def _generate_examples(self, files):
        logger.info("⏳ Generating examples from = %s", files)
        for file in files:
            with open(file, encoding="utf-8") as f:
                guid = 0
                tokens = []
                ner_tags = []
                for line in f:
                    if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                        if tokens:
                            yield guid, {
                                "id": str(guid),
                                "tokens": tokens,
                                "ner_tags": ner_tags,
                            }
                            guid += 1
                            tokens = []
                            ner_tags = []
                    else:
                        # conll2003 tokens are space separated
                        splits = line.split(" ")
                        tokens.append(splits[0])
                        ner_tags.append(splits[1].rstrip())
                # last example
                if tokens:
                    yield guid, {
                        "id": str(guid),
                        "tokens": tokens,
                        "ner_tags": ner_tags,
                    }
