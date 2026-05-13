# Prompt files

| File                                | Purpose                                                       |
|-------------------------------------|---------------------------------------------------------------|
| `vidprom_filtered_extended.txt`     | Training prompts for DMD score distillation (~248k captions). **Not tracked in git** due to its size (~140 MB). |
| `MovieGenVideoBench.txt`            | MovieGen short captions (1k prompts).                         |
| `MovieGenVideoBench_extended.txt`   | LLM-extended MovieGen captions (paired 1:1 with the above).   |
| `vbench/all_dimension.txt`          | VBench evaluation prompts (short form).                       |
| `vbench/all_dimension_extended.txt` | VBench evaluation prompts (extended form).                    |

## Obtaining the training prompts

The 140 MB `vidprom_filtered_extended.txt` file is too large to ship in git. It is
the filtered + extended subset of [VidProm](https://huggingface.co/datasets/WenhaoWang/VidProM)
that we use for self-forcing distillation. Recreate it with:

```bash
huggingface-cli download <your-org>/EndlessWorld-prompts \
    vidprom_filtered_extended.txt --local-dir prompts/
```

(replace the repo with whichever host you upload to — HuggingFace Datasets is
recommended).
