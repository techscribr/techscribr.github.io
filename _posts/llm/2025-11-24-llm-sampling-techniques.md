---
title: "From Boring to Brilliant: A Guide to LLM Sampling Techniques"
date: 2025-11-24 12:00:00 +0530
categories: [Deep Learning]
tags: [LLM]
math: true
---

When you ask an LLM a question, it doesn't just "know" the answer. It predicts the next word, one by one. But *how* it chooses that next word makes the difference between a robotic, repetitive answer and a creative, human-sounding response.

This logic is defined by **Sampling Techniques**. Here is a breakdown of the most common methods, how they work, and when to use them.

---

## **1. The Deterministic Route: Searching for the "Best" Answer**

These methods aim to find the statistically most probable sequence of text. There is no randomness here; the same input will always yield the same output.

### **Greedy Search**
This is the simplest approach. At every step, the model looks at the probability distribution and strictly picks the **#1 most probable token** (argmax).

* **Pros:** It is fast, computationally cheap, and produces very coherent, on-topic text.
* **Cons:** It kills creativity. It is prone to getting stuck in repetitive loops (e.g., "I am I am I am...") and is "brittle"—one bad word choice early on can derail the whole sentence.
* **Best For:** Strictly factual tasks like arithmetic, translation, or Q&A where you want the single "correct" answer.

### **Beam Search**
Think of this as a "smarter" Greedy Search. Instead of only keeping the single best word at each step, it keeps track of the **$k$ most probable sequences** (called "beams") simultaneously.

* **Pros:** It offers high fluency and is better at maintaining overall sentence logic compared to Greedy.
* **Cons:** It is computationally expensive ($k$ times more work) and can still result in generic, safe text.
* **Best For:** Machine Translation and Summarization, where the overall quality of the full sequence is the top priority.

---

## **2. The Stochastic Route: Adding Creativity**

To make text feel human, we need randomness (stochasticity). These methods sample from the probability distribution rather than just maximizing it.

### **Top-K Sampling**
This method truncates the "tail" of the probability distribution. It filters the options to the **Top $k$ most probable tokens** (e.g., the top 50), and then samples randomly from that specific group.

* **Pros:** It balances creativity with coherence by filtering out "absurd" low-probability words.
* **Cons:** It isn't adaptive. The cutoff $k$ is fixed. If the "true" list of good words is 3, but $k$ is 50, you might get a bad word. If the list of good words is 200, you unfairly cut 150 of them.
* **Best For:** Creative tasks like story writing, chatbots, and brainstorming.

### **Top-P (Nucleus) Sampling**
Currently the "gold standard" for open-ended generation. Instead of a fixed number of words, it filters to the smallest set of words whose **cumulative probability is $\ge p$** (e.g., $p=0.92$).

* **Pros:** It is highly adaptive.
    * If the model is certain (one word has 95% probability), the sample pool shrinks to just that word (acting like Greedy).
    * If the model is uncertain, the pool expands to include many options (acting like Top-K).
* **Best For:** The modern default for chatbots (including ChatGPT), coding assistants, and creative writing.

---

## **Deep Dive: Beam Search vs. Top-K Sampling**

A common point of confusion is that both Beam Search and Top-K use a number "$k$." However, they are fundamentally different tools for different jobs.

| Feature | Beam Search ($k=3$) | Top-K Sampling ($k=3$) |
| :--- | :--- | :--- |
| **Type** | **Search** Algorithm (Deterministic) | **Sampling** Method (Stochastic) |
| **Goal** | Find the single, most probable *complete sentence* | Pick one random, "good" *next word* |
| **What is $k$?** | **Beam Width:** Number of parallel sequences to track | **Filter Size:** Number of top tokens to choose from |

### **The "Cat" Example**
Imagine the model is completing the sentence: *"The cat sat on the..."*

**1. How Beam Search handles it ($k=2$)**
Beam search looks at the "Total Score" of the entire sentence so far.
* It calculates 50,000+ possibilities but only keeps the Top 2 paths: "The cat sat on the **mat**" and "The cat sat on the **rug**." It permanently discards "floor" or "dog".
* It then explores the *next* word for those two specific paths to see which yields the highest total probability for the full sentence.
* *Result:* No randomness. It optimizes for the "best" path.

**2. How Top-K handles it ($k=3$)**
Top-K makes a decision for *right now*, with no memory of other paths.
* It takes the top 3 words: **mat** (0.4), **rug** (0.3), **floor** (0.15).
* It recalculates their probabilities to sum to 100% (e.g., mat becomes ~47%, rug ~35%).
* It rolls the dice. There is a real chance it picks "rug" or "floor," even if "mat" was the most likely.
* *Result:* Randomness is introduced. The model creates something new.

---

### **Conclusion**

Choosing the right sampling technique depends entirely on your goal. If you need a mathematically perfect translation, stick to **Beam Search**. If you want a chatbot to write a poem or brainstorm ideas, **Top-P (Nucleus) Sampling** is your best friend.