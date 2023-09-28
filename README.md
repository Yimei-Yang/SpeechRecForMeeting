# Progress Report

Created By: Yimei Yang
Stakeholders: ridhi mittal, Yangjingming Fu, Yimei Yang, Xinyi Zhang, Viv Crowe
Created: June 2, 2022 3:05 PM
Last Edited Time: June 20, 2022 7:54 PM
Last Edited By: Xinyi Zhang

# 💻 Product description

## Problem

A large percentage of the workforce are often overlooked, specifically: introverts, remote workers, and women. Meetings usually consist of a few outspoken members while many don’t get the chance to speak their ideas due to lack of accommodations. 

As a result, those who don’t get a chance to contribute may feel disengaged and demoralized, leading to decreased productivity and hindering the company’s potential growth.

## Solution

The product under development aims to facilitate better communication in meetings. For the scope of this project, we decide to focus on interruption identification in meeting conversations. It will be a plug-in tool for existing virtual meeting platforms (i.e. Zoom, Webex, Microsoft Teams). This tool will utilize the meeting’s recording (i.e. speech audio data and text transcription) to identify when an interruption has occurred, the interrupter, the interruptee, and the topic interrupted. From this data, a report will be generated at the end with suggestions on how to allocate time to highlighted participants and topics. 

# 📁 Data description

In order to give this problem a machine-learning solution, it is important to find a suitable dataset. At first, we broadly looked at many conversational datasets, including AMI, ICSI, SwDA, and some datasets from the research articles studying conflicts in communication. After examining the pros and cons of the potential datasets, we narrowed it down to one corpus: ICSI. 

## ICSI

ICSI is a corpus of 75 meetings (ranging from 17 mins to 103 mins, totaling 72 hours) conducted in English. They were (real) meetings that happened in speech research labs in the early 2000s, with 53 unique speakers. Basic demographic information was also collected for each speaker, including gender, what language they speak, what region they are from, education, position, etc. For all the meetings, what was said was transcribed and the *dialogue acts* were manually labeled. While there exist other corpora that also have dialogue acts annotations, we decided that ICSI’s annotation best suits our needs.

## What is a dialogue act (DA)?

By the definition from the research team that developed this corpus, it is basically the function(s) of an utterance (or a partial utterance). Each annotation of DA contains one or more tags, and each tag represents a function; some indicate the utterance’s general function (e.g., a statement, a question), while some indicate a more specific function (e.g., acknowledging, rejecting). For each DA, there needs to be one and only one general tag; the specific tags are appended to the general tag. There is also a special group of tags called *Disruption Forms.* They can be appended to a general tag, following a `.` or a `|`; it can also stand alone. 

For the purpose of this tool, we will only look at some shortlisted DAs: [Data Investigation](https://www.notion.so/Data-Investigation-5cd4cd95f8854820a324c3444f7b052a?pvs=21) 

| DA Group | Name of the tag | Symbol | Definition | Comments |  |
| --- | --- | --- | --- | --- | --- |
| G12: Disruption Forms | Interruption | %- | Incomplete utterances where a speaker stops talking because they are interrupted by another speaker | disruptive overlapping speech |  |
| G12: Disruption Forms | Abandoned | %-- | Utterances in which a speaker trails off or chooses to reformulate or change the topic by beginning a new utterance (i.e., no change in speaker) | non-overlapping disruption  | Syntactically probably similar to interruption, but not acoustically (no overlapping speech) |
| G4: Backchannels and Acknowledgement | (ALL of them?) | b, bk, ba, bh | Mark utterances that are often responses, in the form of
acknowledgments or backchannels (e.g., showing that one is listening), to a speaker who has the floor. Such responses generally do not elicit feedback and they don’t serve the purpose of halting the speaker who has the floor. | non-disruptive overlapping speech |  |
| G9: Supportive Functions | Collaborative Completion | 2 | Utterances in which a speaker attempts to complete a portion of another speaker’s utterance | supportive overlapping speech |  |
| G11: Further Descriptions  | Topic Change | tc |  Utterances that begin or end a topic  | for topic detection  |  |
| G11: Further Descriptions  | About-Task | t | Utterances that are in reference to meeting agendas or else address the direction of meeting conversations with regard to meeting agendas | for topic detection  |  |

# 📈 Data preprocessing

1. From the  `meetingID.[?].dialogue-acts.xml` files extract dialogue acts of interest (DAoIs). Represent these DAoIs in a dataframe called `df`_diag_acts;
    
    <aside>
    🎈 For our initial model: DAoIs = Interruption
    
    </aside>
    
2. Split the meeting into evenly sized segments. 

```python
def getInputSegmentTimes(audio_file, segment_length, overlap_length) 
```

**Input**: 

- 75 .wav audio_files
- segmentation length
- overlap length

**Output**: a `pd dataframe` called `segments_df`;

- meetingID (str)
- st_time (float)
- ed_time (float)

```python
def getInputSegments(audio_file, df_timestamps)
```

**Input**: 

- 75 .wav audio_files
- pd timestamps generated by the getInputSegmentTimes() function

**Output**: `list` of [`path`1, …, path75] to the segmented audio_files(now in the audio_splitted google drive folder)

• [path1(str)… path75 (str)]

1. Extract `signal` for segments in `audio_splitted` google drive folder 

```python
def extract_features(splitted_path_list, np_list)
```

**Input**: 

- [path1(str)… path75 (str)]
- empty array

**output**: 

- array of `np_arrays` and each of them represents the `melspectrogram` feature extracted from a segment [vector1(np.array)… vector100000(np.array)] **→ This is our** $X$

<aside>
🎈 Later this can include MFCCs and/or other features extracted from the `.wav` files

</aside>

# 🖥️ ML Model

## Architecture overview

### Requirements

The model will

1. Use as input audio data from a meeting.
    - A segment of 10 seconds will be analyzed to determine if an interruption occurs in this time frame.
        
        10 secs is approximately the length of the longest “Interruption” in the Dialogue Acts dataset which motivated this decision.
        
2. Label sections of time in which an interruption occurs.
    1. It will not attribute interruptions to a particular speaker.
3. The model will use

![Technical Research VC - Model architecture.jpg](Progress%20Report%20c77bf6369f4247c9bdc5c1c8e2f150cb/Technical_Research_VC_-_Model_architecture.jpg)

## Evaluation

The following methods and metrics will be reported to evaluate the performance of the model.

1. Training and test error rate, avg from 5-fold cross validation.
2. Number of parameters
3. [ROC/AUC](https://www.youtube.com/watch?v=4jRBRDbJemM)
4. Compare performance between balanced and unbalanced datasets.
5. Inspect 20 examples of false negatives 
6. Inspect 20 examples of false positives

# 📱 Front-end

We’ve designed a draft website mockup on Figma which is inspired from the audio waves we are working on and the key idea of collaboration.

![colors + fonts.png](Progress%20Report%20c77bf6369f4247c9bdc5c1c8e2f150cb/colors__fonts.png)

![landing page wireframe.png](Progress%20Report%20c77bf6369f4247c9bdc5c1c8e2f150cb/landing_page_wireframe.png)