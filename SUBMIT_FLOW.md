# Submit Flow

Deadline: Apr 26 11:59pm PST, which is Apr 27 around 3:00am Peru.

1. Push to GitHub

```bash
gh repo create marimo-comp-2026 --private --source=. --remote=origin --push
```

What to expect: GitHub CLI creates the repo, adds `origin`, pushes the two local commits, and prints the new GitHub URL.

Time estimate: 2-3 minutes.

2. Open molab and import from GitHub

```bash
open https://molab.marimo.io
```

What to expect: molab opens in the browser; choose New notebook, then the Import from GitHub option, and paste/select the GitHub repo URL from step 1.

Time estimate: 3-5 minutes.

3. Verify the notebook runs

What to expect: The notebook loads with the BASIN EXPLORER cell, the train button runs, the sliders update the visualization, and the falsification sweep completes without errors.

Time estimate: 5-8 minutes.

4. Get the molab URL and update README

```bash
MOLAB_URL="PASTE_MOLAB_NOTEBOOK_URL_HERE"
perl -0pi -e "s|MOLAB_URL_PLACEHOLDER|$ENV{MOLAB_URL}|g" README.md
```

What to expect: `README.md` now contains the real molab notebook URL instead of `MOLAB_URL_PLACEHOLDER`.

Time estimate: 2 minutes.

5. Commit and push the README update

```bash
git diff README.md
git add README.md
git commit -m "Add molab notebook link"
git push
```

What to expect: Git shows only the README link replacement, creates one new commit, and pushes it to GitHub.

Time estimate: 2-3 minutes.

6. Submit the competition form

```bash
open https://form.jotform.com/260916218322049
```

What to expect: The form opens; enter name `Railly Hugo / Hunter`, email `railly@clerk.dev`, the molab notebook URL, and optional team info as `solo`.

Time estimate: 5 minutes.

7. Tweet the thread

```bash
open tweet.md
```

What to expect: `tweet.md` opens; paste the molab link into tweet 1, then post the 8 tweets sequentially as a thread.

Time estimate: 5-10 minutes.

8. Optionally post in marimo Discord

What to expect: In the marimo Discord, post the molab notebook link and a short description in the `#molab-competition` channel.

Time estimate: 2-3 minutes.

## Troubleshooting

1. molab cannot import from GitHub

Fallback: Create a New notebook in molab, open `notebook.py` locally, copy the full file contents, and paste them into the new notebook editor manually.

2. molab fails to load

Check whether a dependency is missing from the PEP 723 header in `notebook.py`; if molab/WASM reports a missing package, add it to the header, commit, push, and re-import or refresh.

3. Form submission asks for fields not anticipated

Use the best-guess answer that matches the submission: solo project, molab notebook URL as the main artifact, GitHub repo URL if requested, and the competition notebook as the project description.
