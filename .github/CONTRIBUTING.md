# _(Not only)_ Coding standards

* **README**

    Each experiment has to have it's own README. Here you can find how to write one:
    * [Making READMEs readable](https://open-source-guide.18f.gov/making-readmes-readable/)
    * [README-Template.md by Billie Thompson](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)

* **Python**

    [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/) is in operation.
    
    If you are emacs user, I recommend installing this package: py-autopep8. Configuration:  
    ```elisp
    ;; enable autopep8 formatting on save
    (require 'py-autopep8)
    (add-hook 'elpy-mode-hook 'py-autopep8-enable-on-save)
    ```  
    If you look for the best python/markdown/everything IDE and want to configure it easily, here is a guide for you: https://realpython.com/blog/python/emacs-the-best-python-editor/ and then http://jblevins.org/projects/markdown-mode/ .

* **Git commits**

    * [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/) is in operation.

    * If you work in this repo, remote branch names should follow those templates:

        * Dev branches: `dev/<user name>/<your branch name>`
        * Experiment/Subprojects branches: `exp/<experiment nick>/<branch name e.g. master>`

        Experiments/Subprojects branches will be merged to origin/master after defined milestones.
                
* **Pull requests**

    * If you want to commit to this repo master branch or experiment/subproject branch **create a pull request**.
