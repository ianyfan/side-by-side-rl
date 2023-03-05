# Side-by-side RL

Repository for the page at: https://ianyfan.github.io/side-by-side-rl

Dependencies:
- [Gymnasium](//gymnasium.farama.org)
- [PyTorch](//pytorch.org)

Files:
- Each directory represents one learning algorithm. Within each directory are the following files:
  - `X.py` (where `X` has the same name as the algorithm directory) contains the Python code for the algorithm.
  - `algorithm.tex` contains the LaTeX description of the pseudocode from the paper.
  - `intro.html` contains the introduction for the algorithm shown on the website, written in HTML.
- `common.py` contains code that is used by the algorithms, where the implementation details are not important for the algorithms, and so can be abstracted away, such as the implementation of policy networks and replay buffers.
- `test-algorithm.py` contains a script to test the learning algorithms.
- `requirements.txt` lists the Python libraries required to run the training code.
- `build.py` contains the code for building the website.
- `index-template.html` contains the HTML template for the website, into which the generated HTML for the algorithms are inserted.
- `build-requirements.txt` lists the Python libraries required to run the website-building script.
