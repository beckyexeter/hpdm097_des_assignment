# Meeting 1

**Location:** Cross Keys, St Luke's Campus
**Date:** 24/02/2026
**Time:** 11:00
**Attendees:** Shantanu, Becky

## Discussion

| Agenda Item | Discussion |
| ----- | ----- |
| **Choice of paper to replicate** | We settled on 'A simulation model of bed-occupancy in a critical care unit' by Griffiths et al. as we both have an interest in critical care. |
| **Whether to try and replicate the model ourselves or using AI** | We decided we would like to try recreating the model ourselves to gain more experience and understanding of DES techniques and best practices. However, we also thought it would be interesting to see how the process of using an LLM to code the model would compare to writing the code ourselves. |
| **A logical sequence of steps which could be used to build up the model to the full complexity required to recreate the study results** | We discussed threee iterations that could be used to build up the complexity of the model: <ol><li>A simple model with one type of unplanned admission</li><li>Extending the model to incorporate multiple types of unplanned admissions</li><li>Extending the model to incorporate elective surgery patients</li><li>Extending the model to incorporate ring fencing of emergency and elective beds</li></ol> |
| **Experiments which need to be run to replicate the study results** | We noted four experiments run in the original paper: <ol><li>Increasing the number of beds</li><li>Ring-fencing emergency and elective beds</li><li>Reducing discharge delays</li><li>Changing the scheduling of elective surgery</li><ol> We will focus on running the model with different numbers of beds initially. |
| **Assignment of tasks to be tackled prior to next meeting** | See action items table below. |
| **Location, date, and time of next meeting** | Cross Keys, 03/03/26, 11:00 |

## Action Items

| Task                                                         | Date to be done by | Assigned to |
| ------------------------------------------------------------ | ------------------ | ----------- |
| Code iterations 1 to 3 of the model                          | 03/03/2026      | Becky       |
| Write code to run trials of the model with different numbers of beds | 03/03/2026         | Becky       |
| Plan possible extensions to the experiments in the paper     | 03/03/2026         | Shantanu    |
| Start to develop some LLM prompts for recreating the model | 03/03/2026 | Shantanu |
| Begin work on the introduction and methods sections of the project report | 03/02/2026 | Shantanu |
