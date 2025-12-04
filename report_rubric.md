CS 6220: Project Final Report
Goals: Practice an important job skill: presenting highlights of your work to others.
Resources:
- Report template on Canvas.
Please submit your solution as a single PDF file on Gradescope (see link in Canvas) by the due date and
time shown there. Each group submits only 1 report, making sure all group members are added to the
submission in Gradescope. In general, treat this like a professional report. There will also be point
deductions if the submission is not neat, e.g., it is poorly formatted. (We want our TAs to spend their
time helping you learn, not fixing messy reports or searching for solutions.)
For late submissions you will lose one point per hour after the deadline. This assignment is worth 100
points and accounts for 5% of your final grade. To encourage early work, you will receive a 10-point
bonus if you submit your solution on or before the early submission deadline stated on Canvas. (Notice
that your total score cannot exceed 100 points, but the extra points would compensate for any
deductions.)
As with all software projects, you must include in the root directory of your project a README file that
briefly describes all the steps necessary to build and execute your project. This description should
include all dependencies. For Python projects, create a file named requirements.txt in the project’s root
directory. Each line in this plain-text file specifies a package name and, if needed, version requirements.
For other programming languages, use their equivalent of this approach. The README file will also be
graded, and you will be able to reuse it on all this semester’s assignments with little modification
(assuming you keep your project layout the same). Instructor and TAs must be able to run your project
following the instructions in your README.
Project Report (100 points, incl. supporting files and source code)
This is it: channel all your hard work into this final report by completing the intermediate report with all
your new results.
Deliverables and Important Notes
Submit the report as a single PDF file per group on Gradescope:
1. Make sure the report uses the provided template. The progress report cannot exceed 3 +
2*team-size pages in 11-point font, e.g., a 2-person team cannot exceed 3+2*2=7 pages. (1 PDF
file)
Make sure the following is linked from your report:
2. Files or Jupyter notebooks containing important results discussed in your report.
3. The project in Github Classroom.
If you want to share large (data) files with instructor and TAs, do not add them to your GitHub project.
Instead, do the following:
• Create a new folder in your Northeastern OneDrive and share it with TAs and instructor, but no
one else.
• In that folder, create a subfolder for each homework.
• Copy into that subfolder the big files you want to share.
• Double-check the sharing settings. (By default, files inherit the top-level folder’s permissions.)
The submission time of your project is the latest timestamp of any of the deliverables included. For the
PDF it is the time reported by Gradescope; for the files on GitHub it is the time the files were pushed to
GitHub, according to GitHub. If you want to keep things simple, do the following:
1. Push/copy all requested files to GitHub and make sure everything is there. (Optional: Create a
release with a version number for this snapshot of your repository.)
2. Submit the report on Gradescope. Open the submitted file to verify everything is okay.
3. Do not push any more changes to the files for this HW on GitHub.


CS 6220: Project Report Template
We provide this template for two reasons: (1) It helps you learn a skill—presentation of results—using a
format that tends to work well and gives you structure. This also minimizes point loss due to overlooked
requirements. (2) It helps us grade reports more efficiently, maximizing the value of feedback we can give
you with our limited TA/instructor resources.
For these reasons, it is important that you precisely follow the format, no matter the amount of progress
you made on the different challenges. Not following these requirements can result in point deduction.
You may copy material from your proposal into the final report as needed.
_____________________________________________________________________________________
Team Members and Github Link
State all members of the team.
Also include a link to your Github Classroom repository for the project.
Project Overview
Start with a summary of your project topic and analysis goals. This should be a single short paragraph.
Then present a preview of the highlights of your project results or achievements, again just one or
maybe two short paragraphs.
Think of the overview as follows: You are applying for a job and your future employer is only reading this
section. How can you tell them in about half a page why they should be interested in you and your
project? How can you get them interested to keep reading? (Be confident in your work, but do not
oversell it!)
Input Data
Describe the data you are working with. Include a few data records, if feasible. (Do not include binary
data or lines that are too long.)
Problem
Briefly explain the problem you solved, e.g., “predict label...” or “find clusters of ... to understand ...” or
“find association rules like ...”.
Evidence of Success
For each problem solved, make a concise but strong case for your solution by reporting the most
important outcomes in terms of success measures. For example, for a classification problem select the
appropriate quality measure (e.g., accuracy, AUC, or prediction/recall etc.), briefly state why you picked
it, and report the numbers achieved by your approach. Typically you also want to include at least 1
result demonstrating how you tuned the hyper-parameters of your approach.
Evidence of Meaningfulness
Briefly explain how you checked that the results you obtained are good and meaningful. Focus only on
the most important result(s). Then show evidence to make your case, e.g., by explaining how you
selected individual records and what your inspection of those records reveals about the quality of your
solution.
Conclusions
In one paragraph, state the main achievements of your project and propose 1 or 2 possible next steps
for further improvements that could be explored in future work. This section makes the last impression
on the reader. Think about what you would like that person to remember about your project.
