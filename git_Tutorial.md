# Git 4 dummies

### Why git?

Starting with a brutal copy-and-paste from Wikipedia:

> Git is software for tracking changes in any set of files, usually used for coordinating work among programmers collaboratively developing source code during software development.

In other words, git is a "program" (more precisely a _version control system_) used to keep track of the development of a program/project/whatever.  

To make it more practical, suppose you're developing a new big project that requires hours of coding and trials. So, you start the project, write the first few lines of code and make some tests. Times goes by, and after another long night spent modifying the code, you realize something is wrong since the code have stopped working. You have no idea where the problem is, and you strongly wish you had a backup of the previous working version of the code, before the last (faulty) changes were implemented.  

A rather manual and natural solution to this problem is to make a copy of the project before making any modifications to it, so that you always have a safe backup version of the project. So, you now have a folder `project` containing the project, and a copy of it named `project_copy_1`. For safety, it may be even better to keep track of the whole evolution of the project, so you eventually end up having multiple copies of the project( `project_copy_1`, `project_copy_2`, ...) each one coming from a different day you worked on it.

Well, that's what *git* does. It keeps track of all the changes you make to a file (or set of files), so that you can always revert back to previous version of the file easily. Git does that in a memory efficient way, since it does not copy and store multiple copies of the code, but keeps tracks of only the actual _changes_ you made to it.

End of the historical introduction.
That's mostly all I know about git, and luckily it is sufficient to use it :D

## Using git

### Github
Git can be used both locally on one's personal machine, or collaboratively as is our case. Since the same files need to be accessible at the same time by multiple persons around the globe, these are stored onto an online server "Github" based on git to work. Github just works as a central computer where one can access and share the files with the other collaborators.  
The set of all the files (so the whole project) hosted on Github, is called a *repository*.  

Here is our repository for the Hackathon project: https://github.com/LauraGentini/QRL  

Note that in order to access it, you need to be added as collaborators by Laura, who is the owner.

### Setup
1. *Sign up on Github*   
You need to have an account on Github, so go and sign up to it.

2. *Set SSH Key*  
Then, follow this tutorial to set up a SSH key, which is used to link your account on Github to your local terminal/computer. In this way, you can use your own terminal to make changes to the online repository. The SSH key is a verification mechanism which ensures that you have the rights to do that.
SSH tutorial: https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh

3. *Clone the repository*  
Cloning the repository means that you want to access the online repository on Github, make a local copy of it to your personal computer, and also keep track of the changes that happen. That is, if someone modifies the repository on Github, with just a command you can download all the new changes and synchronize your local copy with the new one. Follow this tutorial to do that: https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository

Now, if all went nice, you should have a folder on your computer called "QRL" (as the name of the repository on Github) containing the files present on the online repository.

### Useful git command
Now that we have a working system, you are ready to make some changes to the code or add some new file. First, you need to check that your local copy of the repository is up-to-date with the one online on Github. To do that, you open your terminal and go to the folder containing the repository:
```bash
cd QRL
```
and then execute the command
```bash
git pull
```
This command says git to go online, check the repository online and see if there is something different with respect to your copy. If so, download them and synchronize.

Then, you do you stuff on your local machine, for example adding a file or modifying a code. After you finish, you want to upload the changes to Github, for everyone to see them. To do that, you need to execute the following commands. First, you tell git which of the changes you made (for example either completely new files, or files which were modified) you wish to upload:
```bash
git add name_of_the_file
```
You repeat that command with all the changes you wish to upload. Once you have finished, you pack all the changes in one single item and you add a brief message explaining what changes it contains:
```bash
git commit -m "a brief message explaining the changes"
```
This is called a *commit*. It is like a checkpoint, summarizing some useful and relevant changes being done to the repository.
Ok, now you are almost done, and we just need to actually send these changes (together with the accompanying message) to the online repository on Github, and you do that with:
```bash
git push
```
Now, you *pushed* your changes online, and everyone can see the new additions/modifications either on Github, or on their terminal in the cloned repository (after they run the `git pull` command to get synced).

If you ever feel lost during any of this stages, you can run the command `git status` which prints some useful information about the changes you made, those you are about to push online, and other.

At last, suppose you made a lot of changes to your local clone of the repository, and you wish to upload them all to Github. Then instead of using `git add` on each of this files, you can use:
```bash
git add .
```
which says to git to include all the new changes in the whole repository.

So, a minimal and complete working example is:
```bash
# go into the repository folder
cd QRL

# check if your version is up-to-date
git pull

# you make some changes to the repository, for example you add a new file
# to it or modify some code already present.

# pick the changes you want to share with others
git add .

# join all files in a single batch and add a comment to the commit
git commit -m "a brief message"

# send the changes to Github
git push

# check the status
git status

# to check it all went fine, you can open the browser, go on Github and check that your changes were actually uploaded
```

These are the most important and probably (and hopefully) the only commands you will need to use.

Of course, there plenty of different scenarios and other commands to use. But it's better to study them case-by-case, since it is already pretty complicated in this way.

That's it. Hope it makes sense.
