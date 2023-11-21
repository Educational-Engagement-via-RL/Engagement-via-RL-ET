# <u>Enhancing Student Engagement through Reinforcement Learning and Eye-Tracking - Proposal</u>
> Note: This document began as a proposal and a process log for the project and will be updated iteratively as the project progresses. The final version will serve as the formal README for the repository. For details on Git, see the [Git Walk Through](https://brook-smash-1a0.notion.site/GitLab-Basics-with-VS-Code-and-DataGrip-An-Introductory-Guide-af772241ddd04660aac5c69784d540fe). It is written for GitLab, but the concepts are analogous for GitHub.

## Table of Contents
1. [Introduction](#introduction)
2. [Key Measurements](#key-measurements)
3. [Model Design](#model-design)
4. [System Workflow for an Individual User](#workflow)
5. [Algorithm](#algorithm)
6. [Result](#result)

## 1. Introduction <a name="introduction"></a>

In the rapidly changing realm of education, personalization is increasingly crucial. While traditional Intelligent Tutoring Systems (ITSs) commendably tailor content to students, they rely on fixed pedagogical rules, potentially lacking the flexibility needed for genuinely individualized learning. Could the dynamism of Reinforcement Learning (RL) and the detailed insights from eye-tracking equipment provide a more adaptive learning experience?

Eye-tracking technology offers insights into a student's cognitive process, revealing their engagement levels, areas of interest, and potential confusion. When paired with RL, which flourished on continuous feedback, there's an opportunity to dynamically adapt educational content based on immediate student needs and reactions. As the eye-tracking system uncovers the student's focus and comprehension, the RL system can promptly adjust its pedagogical method.

In recent years, RL has become increasingly prominent in the pedagogical domain. In this area, the use of RL techniques can be broadly classified into three types: (1) Markov Decision Process-based models, where outcomes are influenced by the decision maker's choices; (2) Partially Observable Markov Decision Processes, which extend MDPs to situations with incomplete state information; and (3) the Deep RL framework, employing neural networks to approximate complex functions such as value and policy functions [1]. Regarding RL research directions in education, they can be categorized into two main categories: (1) Customizing the Procedure of Learning, focusing on methods and strategies in the learning process, including pedagogical approaches, learning paths, and interaction mechanisms; and (2) Customizing the Learning Material, which centers on developing and adapting educational content to meet individual learners' needs. **<span style="background-color: yellow">[Literature Review to be added]</span>**

At the core of most RL techniques and directions mentioned above is the evaluation and processing of a student's engagement level. By combining eye-tracking and reinforcement learning, this project aims to capture immediate student responses and promote learning material that maximizes students' engagement, thereby paving the way for a more personalized learning experience.

It's crucial to recognize that, while this project primarily emphasizes engagement without delving deep into other key educational outcomes, such as content retention or subjective learning experiences, engagement is indeed one of the most crucial metrics in learning, and is positively correlated to students' performance under other metrics. Students deeply engaged in their learning are not only more efficient at content assimilation but also nurture a sustained passion for education. This intrinsic connection to learning positively impacts a spectrum of academic metrics, from retention rates to grasping intricate topics, culminating in heightened overall satisfaction. Thus, by channeling our efforts into maximizing student engagement, we envision crafting a richer, more immersive educational milieu.

Zooming into the specifics of our initiative, we've curated learning materials pivoting around national flags. Given that participants may come from diverse cultural backgrounds, their interest and pre-knowledge in different flags, along with their learning preferences might vary, leading to differing optimal learning policies. Through interaction, the system proposed by this project aims to adapt to each student's learning preference, enhancing their ability engagement in the learning material.

In the subsequent proposal, we will detail the definitions of certain measurements and how we evaluate them, outline the preliminary RL model, and describe the application's workflow.

## 2. Key Measurements <a name="key-measurements"></a>

In educational psychology, significant emphasis is placed on the correlation between a student's engagement level and their familiarity with learning material, as well as the relationship between new learning material and prior knowledge. Theories have developed on how learners' cognitive engagement with instructional material can be influenced by their prior knowledge. Regarding the similarity of learning material, schema theory suggests that if the new material is significantly different, it can be more challenging but also more engaging if appropriately supported. Conversely, according to cognitive load theory, if new material is similar and can be integrated with existing knowledge, it may reduce extraneous cognitive load and foster deeper engagement.

Regardless of the specific impact that a student's familiarity with learning material and the relationship between new learning material and prior knowledge might have on their engagement level, these theories collectively illustrate a correlation between the familiarity and similarity of the material and the student's engagement level.

Besides the levels of similarity and familiarity, students' engagement level is another measurement to be considered and is at the core of this application, as the goal of the algorithm is to maximize students' engagement. Below, we illustrate in detail the evaluation of familiarity, similarity, and student engagement in this project.

### Familarity Level
Students' initial familiarity level with each national flag is based on self-reporting. In the knowledge profiling stage of the experiment, students will be presented with pictures of all the national flags. For each of these flags, students will be asked to rate their familiarity with the flags on a scale from 1 to 3. For more details, please refer to the section 'System Workflow for an Individual User'.

### Similarity Level
**<span style="background-color: yellow">[Illustration of the Model and Simple Data Description]</span>**

**<span style="background-color: yellow">[Simple Data Description]</span>**

### Student's Engagement

In assessing student engagement with educational material, our methodology involves calculating the cumulative heatmap intensity within designated areas of interest. This intensity is derived from Gaussian blur applied to fixation data, utilizing a modified script based on the 'Fixation-Densitymap' repository by yamtak1216 on GitHub. Areas of interest are identified in relation to the flag images or their associated illustrative content.

Given the potential variation in intrinsic engagement levels across different flags, we employ a normalization process for our engagement metrics. This is achieved through the use of a photo-specific engagement coefficient, ascertained via the Intrinsic Image Popularity Assessment [2]. This assessment encompasses a probabilistic approach to generate a substantial dataset of popularity-discriminable image pairs, forming the basis for the pioneering large-scale Intrinsic Image Popularity Assessment (I2PA) database. We leverage computational models, grounded in deep neural networks, to optimize these models for ranking consistency across millions of image pairs. This approach is informed by adaptations from the GitHub repository 'Intrinsic-Image-Popularity' by dingkeyan93.

**<span style="background-color: yellow">[Simple Data Description]</span>**

## 3. Model Design <a name="model-design"></a>

Reinforcement learning (RL) offers a methodical approach to optimizing the decision-making process through interactions with an environment. In the context of this project, the environment is the student's learning experience, and the agent's objective is to personalize this experience by strategically selecting which flags to show, thereby maximizing the student's engagement.

### State
The **state** in our RL model is a tuple of three elements:
- **Student Engagement Level**: The engagement level is deduced from the eye-tracking data and can be classified into three levels: low, medium, and high.
- **Previous Flag Familiarity**: This represents the familiarity category of the last flag shown to the student. It can take values: unfamiliar, not sure, or familiar.
- **Previous Flag Similarity**: A binary variable indicating whether the previously shown flag was similar or not to its predecessor.

> Note: The labels of familiarity — `unfamiliar`, `not sure`, and `familiar` — are pre-defined based on student reports prior to their interaction with the system. The similarity between flags is also predefined. Whether this similarity is determined manually or via a method like a neural network will be finalized as the project advances.

### Action
The agent's **action** is a tuple that consists of:
- **Flag Familiarity**: Deciding which flag familiarity category to present next: unfamiliar, not sure, or familiar.
- **Flag Similarity**: Choosing to present a flag that's either similar or dissimilar to the previous flag.

> Note: Once the familiarity and similarity levels are decided, a flag will be randomly drawn from the pool of flags satisfying the criteria.

### Reward
The **reward** is:
- **Student's engagement level**: After a flag is shown, this engagement can be quantitatively represented using specific numerical values, through data collected via the eye-tracking system. A low engagement level, indicating disengagement or potential confusion, is denoted by a value of -1. A medium engagement level, signifying neutral engagement, is given a value of 0; A high engagement level, reflecting a significant degree of interest or involvement, is marked with a value of 1.

### Expected Policy Relevance
We design the above model with expectations that the optimal policy may be relevant to:
- **Prior Student Engagement**: If a student was highly engaged with a particular flag, the optimal policy might choose a similar flag or one that's progressively more challenging.
- **Familiarity of Previous Flag**: Based on how familiar the student was with the previous flag, the RL model might decide on the next flag's difficulty.
- **Similarity to the Previous Flag**: The relation between consecutive flags might affect engagement.
- **Adaptive Difficulty**: The policy could adapt to the student's performance.

## 4. System Workflow for an Individual User <a name="workflow"></a>

### Stage 0: Knowledge Profiling

In a given trial, we will have a total of 100 flags. Students are presented with a task to rate their familiarity with each flag. They use a scale of 1 to 3, where 3 indicates a clear recognition of the nation associated with the flag, 2 suggests a vague remembrance without certainty of its national association, and `1` denotes unfamiliarity.

Upon completion of the profiling process, the system divides the 100 flags into two distinct groups. Care is taken to ensure both groups have a balanced representation of flags based on familiarity levels. One of these groups is randomly allocated as the test group, where the RL algorithm will be applied. The counterpart group, serving as the control, will experience a learning environment where materials are introduced in a randomized manner.

### Stage 1: Instruction Walk-Through

This phase is designed to acclimatize students to the system's functionalities and workflow with representative flags. In each example, the student will be shown a page with general information about the flag. On the left side of the page, they'll find the flag's visual representation accompanied by a map pinpointing the country's geographical location. Adjacently, the right half of the screen will be populated with textual insights. This includes a brief historical background of the flag and a deeper dive into distinctive patterns characterizing it. The students navigate to the next page via the button `next`.

### Stage 2: Experiments

In this phase, the sequence of appearance for the control group and the test group is randomized. The primary distinction between the test and control groups lies in how the algorithm determines the subsequent flag to display. For the control group, these decisions are made randomly. After each group's learning is finished, a quiz will be initiated to ask students to match flags with the nation names. This quiz is not involved in the RL algorithm but is taken into consideration when deciding whether the algorithm designed to enhance engagement also contributes to other learning outcomes.

### Stage 3: Post-Experiment Interview

At the conclusion of the experiment, we will conduct in-person interviews with each participant to understand their cultural background and gather feedback on their experience with the system. Specifically, we aim to discern any perceived differences between the test and control groups. The detailed questions and structure for these interviews will be provided in a subsequent outline.

## 5. RL Algorithm and Exploration Strategy <a name="algorithm"></a>
For the learning algorithm, we will implement Q-learning, a model-free, off-policy algorithm that seeks to learn the value of an action in a particular state. It does this by iteratively updating action values using a learning rate and a discount factor. To balance exploration (trying new flags) and exploitation (showing flags based on what's been effective), we'll employ an epsilon-greedy strategy. Initially, the agent will mostly explore, but as it accumulates knowledge, it will shift towards exploiting that knowledge. **<span style="background-color: yellow">[To be finalized, add math formula.]</span>**

## 6. Results <a name="result"></a>
**<span style="background-color: yellow">[To be added]</span>**

## Appendix A: References <a name="references"></a>
[1] [Fahad Mon, Bisni, Asma Wasfi, Mohammad Hayajneh, Ahmad Slim, and Najah Abu Ali. 2023. "Reinforcement Learning in Education: A Literature Review" Informatics 10, no. 3: 74.] (https://doi.org/10.3390/informatics10030074).
[2] [Keyan Ding, Kede Ma, Shiqi Wang. 2019. "Intrinsic Image Popularity Assessment" Proceedings of the 27th ACM International Conference on Multimedia] (https://doi.org/10.48550/arXiv.1907.01985)

## Appendix B: UI <a name="UI"></a>
**<span style="background-color: yellow">[To be added]</span>**

## Appendix C: Code <a name="code"></a>
**<span style="background-color: yellow">[To be added]</span>**
