# dpem_code

Hi, thanks for your interests in our DPEM paper! 
Our paper was published at AISTATS 2017, which you can find http://proceedings.mlr.press/v54/park17c.html

This code package consists of several folders
(1) data: which contains Gowalla / lifescience datasets, which are from UCI data repository
(2) pmkt3: contains code copied from pmtk3.googlecode.com, which I made some modifications for adding noise to moments
(3) matfiles: once you run my code, the results will be saved here.

I wrote code for test lifescience data and Gowalla data in Matlab, while my code for DP factor analysis is written in Python.

So, if you want to run "testLifesci.m" or "testGowalla.m", open Matlab. The first thing you want to do is, "startup.m", which generates paths
for running necessary sub-functions. Then run "testLifesci.m" or "testGowalla.m".

If you want to run "DPEM_FA.py", open Python, then run "DPEM_FA.py".

I haven't added my code for testing stroke data, because this dataset isn't publically available. Please contact IMEDS (http://imeds.reaganudall.org/ResearchLab) if you want to get access their data. 

If you have any questions on my implementation, find any bugs, errors, ways to improve, please contact me @ mijungi.p@gmail.com

Have a wonderful day! 

May 19, 2017 @ Amsterdam, Netherlands.
