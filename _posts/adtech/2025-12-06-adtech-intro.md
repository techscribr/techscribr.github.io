---
title: "Introduction to AdTech: The Invisible Auction and its Participants"
date: 2025-12-06 15:00:00 +0530
categories: [AdTech]
tags: [Ad-Tech]
math: true
---

Welcome to the first part of our AdTech primer. For most developers, modern advertising technology is a **black box** of invisible network requests. In this post, we’re cracking it open to understand the high-speed marketplace that powers the internet.

---

## 1. The Problem: Scaling the Manual Deal

In the early days of the web, if a company wanted to advertise, they had to call a website's sales team, negotiate a price, and manually send over image files. While manageable for a few niche sites, this manual approach cannot scale to the billions of pages and trillions of impressions available today.

Modern advertising solves this through **Programmatic (Advertising)** - the use of automated software, data, and algorithms to buy and sell digital ad inventory. It streamlines a process that once took weeks into a workflow that completes under **100-200 milliseconds** to avoid slowing down the user's experience.

### How RTB Relates to Programmatic
While often used interchangeably, it is technically more accurate to view **Programmatic** as the overarching category of automated buying. **Real-Time Bidding (RTB)** is the specific *mechanism* or protocol used within programmatic to conduct an instantaneous auction for each individual impression.

Think of programmatic as the "automation of the transaction", while RTB is the "dynamic price discovery engine". Not all programmatic deals happen via auction as some are fixed-price automated deals, but RTB is what enables the hyper-targeted, high-frequency "spot market" that defines the modern web.

---

## 2. Meet the Players: The Buy and Sell Sides

To understand the auction, we must first understand the platforms that act as the interface for both sides of the market.

### a. The Demand-Side Platform (DSP)

At its core, a DSP allows advertisers to manage their ad campaigns across multiple ad exchanges/marketplaces where publishers make their ad inventory available - all through a single interface. This eliminates the need for manual negotiations with individual websites and provides powerful tools for targeting, bidding and optimization.

The magic happens through **RTB**. In the milliseconds it takes for a webpage to load, an auction takes place for the available ad space. The DSP analyses the data associated with the user visiting the page and decides whether to bid on that ad impression and how much to offer.

* **Core Function:** Analyzes user data (location, history etc.) and page context to calculate the optimal bid for a specific impression.
* **Prominent Players:** 
  * **Google Display & Video 360 (DV360):** The dominant enterprise choice, deeply integrated with the Google ecosystem.
  * **The Trade Desk:** The leading independent DSP, known for its extensive reach across the "open internet" outside of Google.
  * **Amazon DSP:** Leverages Amazon’s massive first-party shopper data to target users based on purchase intent.

#### A Step-by-Step Example: *Urban Hiker* and their New Boot Launch
Let's illustrate this with our hypothetical company **Urban Hiker**, which is launching a new waterproof hiking boot. Their target audience is city-dwelling millennials in India who have an interest in outdoor activities and sustainable fashion.

Here’s how Urban Hiker would use a DSP to run their advertising campaign:

##### Step 1: Campaign Setup in the DSP
The marketing team at Urban Hiker logs into their chosen DSP (popular examples include *Google Display & Video 360*, *The Trade Desk*, *or Amazon DSP*). They create a new campaign and define their objectives: *driving traffic to their product page and increasing online sales*. They set their overall budget, the campaign duration, and upload their ad creatives: eye-catching banners and short video ads showcasing the new boots.

##### Step 2: Defining the Target Audience
This is where the power of the DSP shines. Urban Hiker doesn't just buy ad space on random websites, they target specific individuals. They set up their targeting parameters within the DSP, which can include:
* **Demographics:** Age (25-40), Location (major Indian cities like Mumbai, Delhi, Bangalore).
* **Interests and Behaviours:** Users who have recently searched for "hiking trails near me", "sustainable footwear" or have visited websites related to outdoor gear and travel blogs.
* **Contextual Targeting:** Placing ads on web pages with content about hiking, trekking, and eco-friendly products.
* **Retargeting:** Targeting users who have previously visited the Urban Hiker website but didn't make a purchase. They can even target those who added the new boot to their cart but didn't check out.

##### Step 3: The Real-Time Bidding (RTB) in Action
Now, a potential customer, let's call her Priya, a 30-year-old from Bangalore who has been researching weekend treks, opens a popular travel blog on her laptop. Here's what happens in the background within milliseconds:
1. **Ad Request:** The travel blog's website sends a request to a Supply-Side Platform (SSP), which is the publisher's equivalent of a DSP. The SSP announces that it has an ad impression available and provides anonymized data about Priya (her general location, the type of website she's on, etc.).
2. **Auction Begins:** The SSP passes this information to an Ad Exchange, which acts as a massive digital marketplace. The ad exchange then sends out a bid request to multiple DSPs, including the one used by Urban Hiker.
3. **DSP Analysis and Bidding:** Urban Hiker's DSP receives the bid request and instantly analyses Priya's data. It recognizes that she fits their target audience perfectly: she's in the right location, her Browse history indicates an interest in hiking, and she's on a relevant website. Based on the rules set by the Urban Hiker team, the DSP calculates a bid amount - the maximum they are willing to pay to show their ad to Priya.
4. **Winning the Auction:** The ad exchange receives bids from various DSPs representing different advertisers. The highest bidder wins the auction. Let's say Urban Hiker's DSP placed the winning bid.
5. **Ad Served:** The ad exchange informs the SSP, which then instructs the travel blog's website to display Urban Hiker's ad for the new waterproof boots to Priya.
This entire process, from Priya opening the blog to the ad appearing, happens in the blink of an eye.

##### Step 4: Campaign Optimization and Reporting
The work of the DSP doesn't stop once the ad is shown. It continuously tracks the performance of the campaign. The Urban Hiker marketing team can log into their DSP dashboard at any time and see a wealth of data:
* **Impressions:** How many times their ad has been shown.
* **Clicks:** How many users clicked on their ad.
* **Click-Through Rate (CTR):** The percentage of impressions that resulted in a click.
* **Conversions:** How many users who saw the ad went on to purchase the boots.
* **Cost Per Acquisition (CPA):** How much they are spending on advertising for each sale.

Based on this data, the DSP and the marketing team can make real-time adjustments. If a particular website is delivering a lot of clicks but no sales, they can exclude it from their campaign. If video ads are performing better than static banners, they can allocate more of their budget towards video. This constant optimization ensures that Urban Hiker's advertising budget is being spent as effectively as possible to reach the most relevant audience and achieve their business goals.

In essence, a DSP provides advertisers like Urban Hiker with a powerful and efficient way to navigate the vast and complex digital advertising landscape, ensuring their message reaches the right people at the right time, and at the right price.

### b. The Supply-Side Platform (SSP)

Let's extend the **Urban Hiker** example to understand the crucial role of the **Supply-Side Platform (SSP)**.

If the **Demand-Side Platform (DSP)** is the advertiser's best friend, the **Supply-Side Platform (SSP)** is the publisher's loyal and hardworking agent. They are two sides of the same coin, working together to make the real-time advertising market function.

A publisher is any website or app owner who wants to make money by selling ad space on their property. The SSP is the technology that allows them to do this automatically, efficiently, and at the best possible price.

* **Core Function:** Connects a publisher's site to multiple ad exchanges, sets "Floor Prices" to prevent inventory from being undersold, and ensures "Brand Safety" by blocking undesirable ads. 
* **Prominent Players:** 
  * **Google Ad Manager (GAM):** A hybrid giant that functions as both a publisher ad server and an SSP.
  * **Magnite:** Formed from the merger of Rubicon Project and Telaria, it is the largest independent SSP.
  * **OpenX:** Known for its high-scale global exchange and early leadership in header bidding technology.

#### Back to Our Example: Meet **Wanderlust Weekly**
Let's revisit our scenario. The potential customer, Priya, is Browsing a popular travel blog called **Wanderlust Weekly**. This blog is the publisher. The owner of the blog, let's call him Amit, wants to earn revenue from his content by displaying ads.

However, Amit is a writer, not an ad sales expert. It would be impossible for him to manually negotiate with thousands of potential advertisers like **Urban Hiker**. This is where his SSP comes in.

Here’s how the SSP works from the publisher's (Amit's) perspective:

##### Step 1: Inventory Management via the SSP
Amit has signed up with an SSP (popular examples include *Google Ad Manager*, *Magnite*, or *OpenX*). He has integrated the SSP's code into his **Wanderlust Weekly** website.
Within his SSP dashboard, Amit has control over his ad inventory. He can:
* **Set the Rules:** Amit can define which types of ad spaces are available (e.g., a leaderboard banner at the top, a rectangle ad in the sidebar).
* **Establish Price Floors:** He can set a minimum price for his ad slots. This is a crucial feature that prevents his valuable ad space from being sold for too little. For instance, he might set a floor price of $0.50 CPM (Cost Per Mille, or cost per thousand impressions), ensuring he never receives less than that amount.
* **Manage Brand Safety:** Amit can create blocklists to prevent certain categories of ads (e.g., from his direct competitors, or sensitive categories like gambling) from ever appearing on his travel blog, protecting his brand's reputation.

##### Step 2: Connecting to the Demand
The primary job of Amit's SSP is to connect his available ad inventory to as many potential buyers as possible. The SSP plugs **Wanderlust Weekly** into multiple Ad Exchanges, which are the massive marketplaces where DSPs come to buy.

By connecting to a wide pool of buyers, the SSP creates a competitive environment. More bidders mean higher demand, which drives up the price for Amit's ad space and maximizes his revenue.

##### Step 3: The Real-Time Auction (from the SSP's side)
This is the flip side of the process we saw with the DSP. When Priya lands on the "Wanderlust Weekly" homepage, the following happens from the SSP's point of view:
1. **Impression Becomes Available:** The moment the page starts loading, Amit's SSP recognizes there's an ad impression up for grabs. It gathers the anonymous data about the user (Priya) and the context of the page (a travel blog).
2. **Sending the Bid Request:** The SSP packages this information and sends out a bid request to the ad exchange. It essentially announces: "I have a premium ad spot on a travel blog about to be seen by a user in Bangalore who is interested in outdoor activities. What am I offered?"
3. **Receiving and Evaluating Bids:** The ad exchange broadcasts this request to numerous DSPs. In our example, the DSP for **Urban Hiker** recognizes this as a perfect match and submits a high bid. Other DSPs representing different brands (perhaps another shoe company, a travel agency, or a credit card company) also submit their bids. The SSP receives all these bids in milliseconds.
4. **Selecting the Winner:** The SSP analyses the incoming bids and instantly identifies the highest one - in this case, the bid from Urban Hiker's DSP.
5. **Ad Delivery Confirmation:** The SSP confirms the winning bid and passes the ad creative (the Urban Hiker boot ad) back to Amit's website. The ad is then displayed to Priya as the page finishes loading.

##### Step 4: Revenue and Reporting for the Publisher
Just as the DSP gives advertisers performance data, the SSP provides Amit with a detailed dashboard. He can see:
* **Fill Rate:** The percentage of his ad requests that were successfully filled with an ad.
* **RPM (Revenue Per Mille):** The average revenue he is earning for every 1,000 ad impressions.
* **Top Performing Ad Units:** Which ad slots on his site are making the most money.
* **Demand Partners:** Which advertisers (or their DSPs) are buying the most of his inventory.

This data allows Amit to optimize his content and ad strategy to further increase his earnings.

#### Summary: DSP and SSP Working in Harmony
The digital AdTech ecosystem is a perfect example of a two-sided market, seamlessly connected by technology.
* **The Advertiser** (Urban Hiker) uses a **DSP** to define who they want to reach and how much they're willing to pay. The DSP's goal is to buy the most effective ad impressions at the lowest possible price.
* **The Publisher** (Wanderlust Weekly) uses an **SSP** to manage their ad space and connect to a wide range of buyers. The SSP's goal is to sell ad impressions to the highest bidder to maximize revenue.
These two platforms meet in the middle (the **Ad Exchange**) to conduct a lightning-fast auction, ensuring that the advertiser reaches their target audience and the publisher earns the most money for their valuable content.

### c. The Ad Exchange

The Ad Exchange is the digital trading floor where the DSPs and SSPs meet. It is the technology layer that facilitates the actual real-time auction, receiving bid requests from SSPs and broadcasting them to hundreds of DSPs simultaneously.

* **Core Function:** Matches supply and demand in milliseconds, identifies the highest compliant bidder, and manages the clearing of the transaction.
* **Prominent Players:** 
  * **Google AdX:** The auction engine integrated directly into Google Ad Manager.
  * **Xandr (formerly AppNexus):** Now part of Microsoft, it offers a massive global exchange with sophisticated technical controls.
  * **Index Exchange:** A neutral, independent exchange favored by premium publishers for its focus on transparency and high-quality inventory.

---

## 3. Anatomy of a Millisecond: The RTB Walkthrough

Let’s look at a real-world scenario. **Urban Hiker**, a company launching new waterproof boots, wants to reach city-dwelling millennials in India.

At 12:32 PM, a user named **Priya** in Bengaluru opens a travel blog called *Wanderlust Weekly*. In the time it takes for her page to load, the following happens:

1. **Ad Request:** The blog's website sends a request to the **SSP**. The SSP packages anonymized data about Priya - like her general location and the fact she's reading a travel blog.
2. **The Auction Opens:** The SSP passes this info to the **Ad Exchange**, which broadcasts a "bid request" to multiple DSPs.
3. **The Decision:** Urban Hiker’s **DSP** receives the request. It recognizes Priya as a high-value target because of her browsing history and location.
4. **Bidding:** Based on pre-set (ML/heuristic) algorithm, the DSP calculates the maximum it’s willing to pay (e.g. ₹7.50) and submits a bid.
5. **Selecting the Winner:** The Ad Exchange receives bids from various advertisers and awards the slot to the highest bidder.
6. **The Reveal:** The ad is served, and Priya sees the new Urban Hiker boots on her screen.

![Image of RTB](assets/img/ad-tech/rtb-flow.png)

---

## 4. Understanding Metrices on Both Sides
With the SSP acting as the publisher’s gatekeeper and the DSP as the advertiser’s strategic buyer, the technical machinery for the auction is complete. But in this high-frequency market, how do both sides keep score? Because the goals of the advertiser/buyer (minimizing cost) and the publisher/seller (maximizing yield) are inherently different, the industry uses a specific set of standardized metrics to translate these competing interests into a common language of value.

In this section, we’ll break down the "Big Four" metrics — CPM, eCPM, ROAS and RPM, to see how they connect in a real-world campaign.

#### CPM (Cost Per Mille)
* **Definition:** CPM is a pricing model used by advertisers to buy ad inventory. An advertiser agrees to pay a fixed price for every 1,000 impressions of their ad. It is a way to buy media.
* **Perspective:** Advertiser
* **Formula:** Total Cost = (Total Impressions / 1000) * CPM Rate
* **Example:** Urban Hiker agrees to pay a news website a CPM of ₹150. If the ad is shown 2,000,000 times, the total cost is fixed and known upfront: (2,000,000 / 1000) * ₹150 = ₹300,000.

#### eCPM (Effective Cost Per Mille)
* **Definition:** eCPM is a performance metric used by advertisers to measure the effectiveness of their campaigns, especially those not bought on a CPM basis (e.g., Cost Per Click - CPC). It translates the performance of any campaign into a common currency: the effective cost per 1,000 impressions. It is a way to analyze campaign cost-effectiveness.
* **Perspective:** Advertiser
* **Formula:** eCPM = (Total Cost / Total Impressions) * 1000
* **Example:** Urban Hiker runs a campaign on a travel blog, paying ₹10 per click (CPC). The campaign gets 500 clicks from 50,000 impressions.
  * Total Cost = 500 clicks * ₹10/click = ₹5,000.
  * Urban Hiker's eCPM = (₹5,000 / 50,000) * 1000 = ₹100.
  * This allows Urban Hiker to compare the cost of this CPC campaign directly against other campaigns.

#### ROAS (Return on Ad Spend)
* **Definition:** ROAS is a primary performance metric used by advertisers to measure the gross revenue earned for every unit of currency spent on advertising. It answers the fundamental question: *"For every ₹1 I put into this campaign, how many rupees did I get back?"*.
* **Perspective:** Advertiser.
* **Formula:** `ROAS = Total Revenue Generated from Ad Campaign / Total Cost of Ad Campaign`.
* **Example:** **Urban Hiker** spends ₹1,15,000 on a campaign for their new trekking jackets. The campaign generates ₹4,50,000 in total sales.
* **Urban Hiker's ROAS** = ₹4,50,000 / ₹1,15,000 = **3.91**.
* **Interpretation:** This is expressed as a **3.9:1 ratio**, meaning every ₹1 spent generated ₹3.90 in revenue.

##### How ROAS Relates to GMV

In the world of e-commerce and retail AdTech, **GMV (Gross Merchandise Value)** is the total sales value for merchandise sold through a platform over a specific period.

The relationship is straightforward. The "Total Revenue" used in the ROAS formula is the portion of the GMV that can be directly attributed to the ad campaign, i.e. the **Attributed GMV**. If a user clicks an ad and buys a ₹2,500 jacket, that ₹2,500 is added to the campaign's attributed GMV.

* **The Calculation:** `ROAS = Attributed GMV / Ad Spend`.

Essentially, ROAS measures the **efficiency** of turning ad spend into GMV. While a high ROAS indicates a successful top-line revenue driver, advertisers must balance it against profit margins. If the cost of goods and operations exceeds the revenue generated, even a high ROAS might not result in a positive **ROI (Return on Investment)**.

#### RPM (Revenue Per Mille)
* **Definition:** RPM is a performance metric used by publishers (the website or app owners) to measure their ad earnings. It shows how much revenue they make for every 1,000 impressions shown on their site, regardless of how the ads were sold.
* **Perspective:** Publisher
* **Formula:** RPM = (Total Earnings / Total Impressions) * 1000
* **Example:** A travel blogger shows ads on their website and earns a total of ₹7,000 in a month from 200,000 impressions.
  * The blogger's RPM = (₹7,000 / 200,000) * 1000 = ₹35.
  * This means for every 1,000 impressions they served, they earned ₹35.

### The Connection: A Unified Example
Let's trace a single campaign to see how all three terms relate.

#### The Setup
* **Advertiser:** Urban Hiker
* **Publisher:** A popular travel blog
* **Intermediary:** An ad network that takes a 30% fee.

#### The Campaign
* **Urban Hiker** (Advertiser) runs a CPC campaign on the blog, bidding ₹10 per click. The campaign generates 500 clicks from 50,000 impressions.
* **Urban Hiker's Analysis (eCPM):**
  * Urban Hiker's total cost is 500 clicks * ₹10/click = ₹5,000.
  * Urban Hiker's eCPM = (₹5,000 / 50,000 impressions) * 1000 = ₹100.
  * Urban Hiker's takeaway: "Our effective cost for this campaign was ₹100 per thousand impressions."
* **The Publisher's Earnings (RPM):**
  * The ad network takes its 30% fee from Urban Hiker's spend: 0.30 * ₹5,000 = ₹1,500.
  * The publisher's total earnings are ₹5,000 - ₹1,500 = ₹3,500.
  * The publisher's RPM = (₹3,500 earnings / 50,000 impressions) * 1000 = ₹70.
  * Publisher's takeaway: "For every thousand impressions I showed, I earned ₹70."
* **The Business Result (ROAS & GMV)**
Now, let’s look at what happened *after* the click. Suppose out of those 500 clicks, **10 people** bought a pair of "Urban Hiker" boots at **₹2,500 per pair**.
  * **GMV (Gross Merchandise Value):** 10 pairs * ₹2,500 = **₹25,000**. This is the total revenue attributed to the ad campaign.
  * **Urban Hiker's Analysis (ROAS):** ₹25,000 (Revenue) / ₹5,000 (Ad Spend) = **5.0**.
  * *Takeaway:* "For every ₹1 we spent on this travel blog, we generated ₹5 in sales (a 5:1 ROAS)."

### **What’s Next?**

We’ve seen how the SSP, DSP work at a high level. In our [next](https://techscribr.github.io/posts/publisher-advertiser-ad-servers/) article, we’ll dive into the internal architecture of these platforms - specifically how **Ad Servers** and **SSPs** manage inventory and *Direct Deals* versus *programmatic auctions*.
