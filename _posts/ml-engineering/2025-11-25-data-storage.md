---
title: "A Short Primer on Scalable Data Storage"
date: 2025-11-25 12:00:00 +0530
categories: [ML Platform]
tags: [ML-Platform]
math: true
---

## Provide a consolidated view of popular data storage technologies relevant in ML based Ecosystems. 

Here is a consolidated view of the most popular data storage technologies used in modern system design. It's grouped by their primary architectural category to make the comparison easier.

### **1. Relational Databases (RDBMS) - The General Purpose Workhorses**

These are the default choice for structured data where data integrity is paramount.

| Technology | **PostgreSQL** |
| :--- | :--- |
| **Type** | SQL (Relational) |
| **Pattern** | **OLTP** (Online Transaction Processing) |
| **ACID** | **Yes** (Strict ACID compliance) |
| **Scalability** | **Vertical** (Scale Up). Horizontal scaling is hard (requires sharding/read replicas), but tools like Citus exist. |
| **CAP Theorem** | **CA** (In practice, it prioritizes Consistency and Availability, usually single-master). |
| **Primary Use Case** | Core application DB, User profiles, Financial ledgers, Complex joins. |
| **Latency** | Low (ms). |
| **Managed Service** | Yes (AWS RDS, Google Cloud SQL, Azure Database). |
| **Critical Info** | The industry standard for general-purpose app development. Very extensible (PostGIS for geo, JSONB for NoSQL-like features). |

### **2. Data Warehouses (OLAP) - The Analytics Engines**

These are designed for scanning billions of rows to answer business questions. They are *not* for live application backends.

| Technology | **Google BigQuery** | **Amazon Redshift** | **Apache Hive** |
| :--- | :--- | :--- | :--- |
| **Type** | SQL (Columnar) | SQL (Columnar) | SQL-like (HQL) on Hadoop |
| **Pattern** | **OLAP** (Online Analytical Processing) | **OLAP** | **OLAP** |
| **ACID** | Atomic updates supported, but not designed for transactional work. | Yes (Snapshot isolation). | ACID support is available (ACID v2) but limited compared to RDBMS. |
| **Scalability** | **Horizontal** (Massively Parallel Processing - MPP). | **Horizontal** (MPP). | **Horizontal** (Uses HDFS/Cloud Storage). |
| **Use Case** | Ad-hoc analysis, BI dashboards, ML training data generation. | Enterprise Data Warehouse, BI reporting. | Batch processing on huge datasets (Petabyte scale) over HDFS/S3. |
| **Latency** | Seconds to Minutes (High). | Seconds to Minutes (High). | Minutes to Hours (Very High). |
| **Managed?** | **Serverless** (Fully Managed). | **Cluster-based** (You manage nodes) or Serverless. | Often self-managed or via EMR/Dataproc. |
| **Critical Info** | Decouples compute and storage perfectly. You pay per query. | Tightly coupled compute/storage (traditionally), though RA3 nodes fix this. | Legacy tech. Being replaced by Spark SQL or Presto/Trino. |

### **3. NoSQL: Key-Value & In-Memory - The Speed Demons**

Used for caching, real-time features, and session management.

| Technology | **Redis** | **Aerospike** |
| :--- | :--- | :--- |
| **Type** | NoSQL (Key-Value) | NoSQL (Key-Value) |
| **Pattern** | **OLTP** (Real-time) | **OLTP** (Real-time at Scale) |
| **ACID** | Atomic single-key ops. No multi-key transactions. | Strong Consistency available (ACID for single row). |
| **Scalability** | Horizontal (Redis Cluster) - **AP** or **CP** depending on config. | **Horizontal** (Linear scale). **AP** (Available/Partition Tolerant). |
| **Use Case** | Caching, Session Store, Leaderboards, Pub/Sub. | Real-time Feature Store (Fraud detection), Ad-Tech bidding. |
| **Latency** | **Microseconds** (In-memory). | **Sub-millisecond** (Hybrid RAM/SSD). |
| **Managed?** | Yes (AWS ElastiCache, Redis Enterprise). | Yes (Aerospike Cloud). |
| **Critical Info** | Single-threaded event loop. Data must usually fit in RAM. | Optimized for Flash/SSD. Can store TBs/PBs of data with RAM-like speed. |

### **4. NoSQL: Wide-Column Stores - The Big Data Writers**

Designed for massive write throughput and storing petabytes of data.

| Technology | **Apache Cassandra** | **Google Bigtable** | **Apache HBase** |
| :--- | :--- | :--- | :--- |
| **Type** | NoSQL (Wide-Column) | NoSQL (Wide-Column) | NoSQL (Wide-Column) |
| **Pattern** | **OLTP** (Heavy Write) | **OLTP** (Heavy Write/Read) | **OLTP** (Heavy Write) |
| **ACID** | No (Tunable Consistency). | No (Single-row atomicity). | Strong Consistency (Single row). |
| **Scalability** | **Horizontal**. **AP** (Eventual Consistency). | **Horizontal**. **CP** (Strong Consistency). | **Horizontal**. **CP** (Strong Consistency). |
| **Use Case** | IoT sensor data, activity logs, messaging apps (Discord/Messenger). | Time-series data, financial data, user history (Gmail/Search/Maps). | Hadoop-native random access storage. |
| **Latency** | Low (Single digit ms). | Low (Single digit ms). | Low to Medium. |
| **Managed?** | Yes (AWS Keyspaces, Datastax). | **Fully Managed** (Serverless). | Self-managed or Cloud Bigtable. |
| **Critical Info** | "Masterless" architecture (no single point of failure). Great for writes. | Powered by Colossus (Google's file system). Storage/Compute decoupled. | Built on top of HDFS. Requires ZooKeeper. High operational overhead. |

### **5. NoSQL: Document & Search - The Flexible Ones**

Used for unstructured data and complex text searching.

| Technology | **MongoDB** | **Elasticsearch** |
| :--- | :--- | :--- |
| **Type** | NoSQL (Document - JSON) | NoSQL (Search Engine / Document) |
| **Pattern** | **OLTP** | **OLAP** (Text Analytics / Log Search) |
| **ACID** | Yes (Multi-document ACID supported since v4.0). | No (Near real-time, eventual consistency). |
| **Scalability** | **Horizontal** (Sharding). **CP** (Consistency/Partition Tolerance). | **Horizontal**. **AP** (Focuses on Availability). |
| **Use Case** | Content management, Catalogs, Rapid prototyping (Schema-less). | Full-text search, Log analytics (ELK Stack), Autocomplete. |
| **Latency** | Low (ms). | Low (ms) for search, Slower for updates. |
| **Managed?** | Yes (MongoDB Atlas). | Yes (Elastic Cloud, AWS OpenSearch). |
| **Critical Info** | Very developer-friendly. Flexible schema allows rapid iteration. | Based on Lucene. Not a primary datastore (danger of split-brain). |

### **Summary of CAP Theorem Categorization**

* **CP (Consistency + Partition Tolerance):** MongoDB, HBase, Bigtable, Redis (if configured for strong consistency), Kafka.
    * *Trade-off:* If the network partitions, the system may stop accepting writes to ensure data doesn't diverge.
* **AP (Availability + Partition Tolerance):** Cassandra, DynamoDB, Couchbase, Aerospike (in AP mode).
    * *Trade-off:* If the network partitions, the system stays up, but you might read stale data ("Eventual Consistency").
* **CA (Consistency + Availability):** RDBMS (Postgres, MySQL).
    * *Trade-off:* They cannot handle network partitions well. They usually run on a single node or a primary/replica setup where partition tolerance is sacrificed (the system goes down or becomes read-only).

### **Summary Table for Quick Reference**

| Name | Category | ACID | CAP | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **Postgres** | RDBMS | Yes | CA | Relational data, Core Apps |
| **BigQuery** | Data Warehouse | No | - | Analytics on Petabytes |
| **Redis** | Key-Value | Atomic | CP | Caching, Sessions, Queues |
| **Aerospike** | Key-Value | Atomic | AP | Real-time Feature Store (Fraud) |
| **Cassandra** | Wide-Column | No | AP | Massive Write workloads (IoT) |
| **Bigtable** | Wide-Column | Atom | CP | High throughput K-V at scale |
| **Elastic** | Search | No | AP | Text Search, Log Analysis |
| **Mongo** | Document | Yes | CP | Flexible Schema, Catalogs |

---

## What's the difference between 'Columner' and 'Wide-Column' data storage solutions?

That is an excellent question that clarifies a very common confusion. Although both terms contain the word "column," **columnar** and **wide-column** storage systems are built for completely different purposes and have vastly different internal designs.

Here is a side-by-side comparison.

### Columnar Storage (The OLAP/Analytics Engine)

Columnar storage systems are built to run **analytical queries (OLAP)** by prioritizing **reading efficiency** over everything else.

#### Core Idea: Storage by Attribute

* **How it Works:** Data is stored on disk grouped by **attribute (column)**, not by row.
    * Example: A table with columns `(User_ID, Name, Sale_Amount, Date)`. All `Sale_Amount` values are stored next to each other on the disk, and all `Date` values are stored next to each other.
* **Best For:** **Read-heavy, analytical queries** that involve aggregating data over large numbers of rows.
* **Key Benefit (I/O Reduction):** When you run a query like `SELECT AVG(Sale_Amount)`, the database only reads the `Sale_Amount` column from the disk. It completely skips reading the `User_ID`, `Name`, and `Date` columns, which drastically reduces I/O and speeds up the query.
* **Data Types:** Optimized for homogeneous, structured data (like SQL tables).
* **Examples:** **Google BigQuery**, **Amazon Redshift**, Snowflake.

### Wide-Column Stores (The Scalable OLTP Engine)

Wide-column stores are a **NoSQL** database design built to handle **massive write throughput and petabyte-scale horizontal scaling**. They are designed for operational workloads.

#### Core Idea: Flexible, Distributed Rows

* **How it Works:** Data is stored as rows, but each row can have a massive number of columns (hence "wide"). Unlike a traditional RDBMS, columns can be created on the fly, and different rows in the same table can have different sets of columns. The columns are grouped into structures called "Column Families" or "Bins."
* **Best For:** **Extremely high-throughput writes** (like log ingestion, sensor data, or storing user history) and **fast key-based lookups**.
* **Key Benefit (Scalability & Flexibility):** The database automatically partitions rows across thousands of machines based on the primary key, providing linear scalability for writes and reads. The dynamic schema allows rapid application development.
* **Data Types:** Optimized for mutable, semi-structured data where the schema is expected to evolve.
* **Examples:** **Apache Cassandra**, **Google Bigtable**, **Apache HBase**, Aerospike (conceptually shares this distributed, K-V model).

### Side-by-Side Comparison

| Feature | Columnar Storage | Wide-Column Stores |
| :--- | :--- | :--- |
| **Primary Goal** | **Analytics (OLAP)** | **Scalable Transactions (OLTP/Real-Time)** |
| **Data Layout** | **Data is stored by column.** (All `Sale_Amount` values are adjacent). | **Data is stored by row** (but rows are wide and flexible). |
| **Optimization** | **Read I/O.** Only read the columns you need. | **Write Throughput** and **Horizontal Partitioning**. |
| **Schema** | Fixed (Schema-on-Write). | Flexible/Dynamic (Schema-on-Read). |
| **Typical Query** | `SELECT SUM(Sales) GROUP BY Month` | `GET user_data WHERE user_id = 123` |
| **Key Advantage** | Massive cost and speed savings on analytical queries. | Linear scalability and predictable low latency for key lookups. |
| **Examples** | BigQuery, Redshift, Snowflake | Cassandra, Bigtable, ScyllaDB |


## Do Wide-column stores allow partitioning and indexing? If so, then explain how.

**Wide-column stores absolutely allow both partitioning and indexing.** In fact, these features are essential to their core function of providing massive horizontal scalability and fast retrieval.

The difference lies in **how** they implement them compared to a traditional SQL database.

Here is a breakdown of how partitioning and indexing work in a wide-column store (using Cassandra/Bigtable as primary examples).

### 1. Partitioning (The Key to Scale)

Partitioning is the primary mechanism for scaling wide-column stores horizontally. It dictates **where** data lives on the cluster.

#### How it Works

* **Partition Key (Required):** When you define a table, you must designate one or more columns as the **Partition Key** (or Row Key in Bigtable).
* **Data Distribution:** The database uses a **hashing function** on the Partition Key to determine which physical machine (node) in the cluster will store that data.
* **Data Grouping:** All rows (records) that share the **same Partition Key** are guaranteed to be stored together on the **same physical node**.

#### The Impact

* **Fast Writes/Reads:** Queries that target a specific partition key (e.g., `SELECT * FROM users WHERE user_id = 'Alice'`) are extremely fast because the system knows exactly which single node to query, avoiding a cluster-wide search.
* **Scalability:** If your cluster has 100 machines, partitioning ensures your 100 terabytes of data are evenly distributed across all 100 machines, allowing the database to scale linearly.
* **The Limitation:** You **must** know the Partition Key to query efficiently. Trying to query without it results in an expensive "full cluster scan," which is discouraged.

### 2. Indexing (The Key to Flexibility)

Indexing in wide-column stores is available, but it is implemented differently than in an RDBMS, and it often comes with significant performance trade-offs.

#### The Problem with Traditional RDBMS Indexes

In a traditional RDBMS, if you index the `city` column, the index is a separate structure that maps all `city` values to the exact physical location of the rows on disk. This is fast for lookups but complex to update.

#### The Wide-Column Approach: Secondary Indexes

Wide-column stores use **Secondary Indexes** to allow queries on non-primary-key columns.

* **Primary Access (The "Main Index"):** The **Partition Key** acts as the primary, built-in index, optimized for that specific query.
* **Secondary Indexes:** These are separate, secondary tables that the database creates and manages automatically.
    * **How it Works (Conceptual):** If you create a secondary index on the `status='active'` column, the database effectively creates a new internal key-value table:
        * `Key: 'status_active'`
        * `Value: [user_id_1, user_id_50, user_id_300, ...]`
* **Trade-off (The "Cost"):**
    * **Reads (Pros):** They allow you to run queries like `SELECT * FROM users WHERE city = 'New York'`, which would otherwise require a full cluster scan.
    * **Writes (Cons):** Every time you update the `city` column on a user record, the database has to also update the massive, distributed **Secondary Index table**. This adds latency to your write operations, which defeats the purpose of using a high-throughput system like Cassandra.

**Conclusion:**

Both **Partitioning** and **Indexing** are used in wide-column stores. **Partitioning** is your mandatory, structural mechanism for scalability. **Indexing** is an optional feature (the secondary index) that provides query flexibility but adds significant overhead to your critical write path and must be used sparingly.

---


## Show examples of some 'good' and 'bad' queries for storage systems like BigTable.

This is a fantastic way to understand the limitations and design philosophy of Bigtable. Unlike a relational database, where a simple `SELECT` can be slow due to poor indexing, Bigtable's speed is entirely governed by how efficiently the system can locate and scan data using the **Row Key**.

Here is a list of three fast and three very slow queries on Bigtable.

### Fast Queries (O(1) to O(logN) by Row Key)

The fast queries are all based on **Row Key lookups** or **contiguous Row Key range scans**. Bigtable is optimized to find data based on the key because the key maps directly to a specific server node in the cluster.

| # | Query Goal | Row Key Structure | Why It's Fast |
| :--- | :--- | :--- | :--- |
| **1** | **Single Point Lookup** (Finding one user's profile) | `user_4321` | This is the fastest possible operation. Bigtable hashes the key and immediately sends the request to the *one server* that owns that row. It's an **O(1)** lookup. |
| **2** | **Time-Series Range Scan** (Recent history for one device) | `device_XYZ#2025-11-19-14:00:00` | The user queries for `key_prefix='device_XYZ' AND time BETWEEN T1 AND T2`. Since all data for one device is stored **contiguously** on a few servers, Bigtable only scans a small, optimized range. This is often called a **Row Key Prefix Scan**. |
| **3** | **Small Transaction Lookup** (A tiny slice of the data) | `transaction_1234` | Similar to #1, this is a direct, targeted read. If your key design is good (e.g., you've structured the key as `user_ID#transaction_ID`), you still benefit from the fast primary lookup. |

### Slow Queries (Full Cluster Scan / Scatter-Gather)

The slow queries are those that force Bigtable to abandon the Row Key and **scan large portions of the cluster** to find the answer. This is the **anti-pattern** for Bigtable.

| # | Query Goal | Reason for Slowness | Why It's Slow (Full Scan) |
| :--- | :--- | :--- | :--- |
| **1** | **Secondary Attribute Lookup** (Finding all users in one country) | `SELECT * WHERE country_code = 'AUS'` | **Bigtable has no secondary indexes.** To answer this, it must read the `country_code` column of *every single row* on *every single server* in the cluster. This is the definition of an expensive, slow, **Full Cluster Scan**. |
| **2** | **Complex Aggregation** (Counting the total number of items) | `COUNT(*)` or `GROUP BY` on a non-key column. | Bigtable must read every single row in the cluster and then perform a cluster-wide aggregation. While faster than traditional databases, it is the slowest possible operation in Bigtable and defeats its design goals. |
| **3** | **Poorly Designed Key Range Scan** (Scanning a non-contiguous range) | `user_ID LIKE 'A%'` (If users A-Z are scattered) | If the users whose IDs start with 'A' are stored across 50 different servers (i.e., they are not contiguous on disk), the query requires 50 parallel network requests, making the operation slow and high-latency. **Effective range scans require careful Row Key design.** |

### The Core Principle

To use Bigtable correctly, you must ensure that **all your read queries are satisfiable by the Row Key and its prefix.** If you find yourself needing to query based on an attribute that is *not* part of the Row Key, you should use a different database (like **Elasticsearch** or **BigQuery**).