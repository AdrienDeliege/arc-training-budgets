# Publishing this package

This folder is a standalone repository candidate. It contains no private Git history, account tokens, cluster logs, training checkpoints, or manuscript submission conversations.

Before publishing, run:

```sh
python scripts/verify_package.py
python -m unittest discover -s tests -v
```

The draft uses repository name `arc-training-budgets`. Once the authors choose the destination and approve public publication:

```sh
git init -b main
git add .
git commit -m "Prepare VARC/TRM training-budget study release"
gh repo create YOUR_GITHUB_ACCOUNT/arc-training-budgets --public --source . --remote origin --push
git tag -a v1.0.0 -m "BNAIC/BeNeLearn 2026 code release"
git push origin v1.0.0
gh release create v1.0.0 --title "BNAIC/BeNeLearn 2026 code release" --notes-file docs/RELEASE_NOTES.md
```

The repository destination above is a proposed name, not an existing verified release. Replace the paper's acceptance-time promise with a link only after the public repository is accessible. Keep the existing private research repositories private; publish this isolated folder.

A Linux/CUDA end-to-end diagnostic should be run in each pinned environment before labelling the release GPU validated. Full historical scores have not been rerun during packaging. If trained checkpoints or raw predictions are added later, include their source configuration, validation-selection criterion, hashes, and coverage in a separate artifact manifest.
