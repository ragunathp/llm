Application Servers	Session Replication	Replicates user sessions across multiple servers for redundancy and failover.	Maintain user session continuity during server failure.	Can increase latency due to replication.	Simulate server failure and verify session continuity.
                    Load Balancing	Distributes requests across multiple application servers to avoid overloading any single server.	Enhance scalability and high availability.	Load balancing misconfiguration can lead to uneven load.	Simulate varying traffic to ensure even load distribution.
                    Graceful Shutdown	Ensures ongoing requests complete before shutting down a server.	Avoid service interruptions during server shutdown.	Can impact deployment speed if not managed carefully.	Test shutdown scenarios to ensure proper handling of ongoing requests.


Database	Database Replication	Maintains redundant database copies to ensure high availability.	Ensure continuous database availability.	Can lead to data consistency issues.	Test failover scenarios to ensure data continuity.
          Database Clustering	Distributes database load across a cluster of nodes for scalability and redundancy.	Enhance scalability and high availability.	Complexity in setup and potential cluster communication issues.	Simulate node failures and monitor load distribution.
          Read Replicas	Uses read-only replicas to offload read operations from the primary database.	Reduce load on the primary database and improve performance.	Added latency due to replication lag.	Measure replication lag and check for data consistency.
          Automated Backups	Schedules regular database backups to prevent data loss.	Ensure data recoverability in case of failure or corruption.	Requires additional storage and can impact performance.	Test backup and restore processes to ensure data integrity.

Message Bus	Redundant Brokers	  Ensures multiple message brokers for redundancy and failover.	Ensure message bus reliability and availability.	   Increased infrastructure cost.    Simulate broker failure and check message delivery continuity.
	          Persistent Messaging	 Stores messages persistently to prevent data loss during failure.	Guarantee message durability and prevent loss.	Additional storage and overhead.	Force broker restart to ensure no message loss.
Kafka	Topic Replication	Replicates Kafka topics across multiple brokers for redundancy.	Ensure high availability and data redundancy.	Increased storage and network traffic.	Simulate broker failure and check data availability.             
            Partition Rebalance	Redistributes partitions for load balancing and fault tolerance.	Ensure even load distribution and fault tolerance.	Potential temporary service disruption during rebalance.	Force partition migration to ensure smooth rebalancing.
            Resiliency Strategy	Description	Objective	Points to Pay Attention	Validation Method
            Topic Replication	Replicates topics across multiple brokers.	Ensures data availability and fault tolerance in case of broker failure.	Higher storage and network requirements.	Simulate broker failure to check data availability.
            Broker Redundancy	Configures multiple brokers in a cluster.	Provides redundancy and load distribution.	May require careful configuration to avoid imbalance.	Test with different broker failure scenarios.
              Controller Redundancy	Configures multiple controller nodes for broker cluster.	Ensures high availability in case of controller failure.	Controller failover may cause temporary instability.	Test controller failover and observe cluster stability.
              Partition Reassignment	Redistributes partitions across brokers for load balancing and fault tolerance.	Prevents uneven load distribution and ensures failover capability.	Reassignment can cause temporary service disruption.	Simulate broker failure and check partition distribution.
              Persistent Storage	Uses durable storage to retain messages and ensure they are not lost.	Prevents message loss due to broker failure.	May require additional storage capacity.	Simulate broker restart to ensure message persistence.
              Log Compaction	Allows Kafka to compact logs to remove duplicate keys, ensuring data consistency.	Ensures data consistency and reduces storage requirements.	May impact performance during compaction.	Test with high volumes of compaction to ensure stability.
              Rack Awareness	Configures brokers with knowledge of physical rack locations for better fault tolerance.	Ensures high availability even with hardware failures in specific locations.	Requires additional configuration to ensure proper distribution.	Test failover scenarios involving hardware failure in specific racks.
              Quotas and Throttling	Sets limits on resource usage for producers and consumers to avoid overload.	Prevents resource exhaustion and ensures system stability.	Overly restrictive quotas may impact performance.	Simulate high load to ensure quotas and throttling function as expected.
              Multi-Cluster Replication	Configures Kafka Connect or other tools to replicate data between multiple Kafka clusters.	Ensures data availability and disaster recovery across clusters.	Requires additional infrastructure and network resources.	Test replication across clusters and validate data consistency.
              Access Control Lists (ACLs)	Defines permissions for producers, consumers, and other Kafka clients.	Ensures security and prevents unauthorized access.	ACL misconfigurations can cause service disruption.	Test with various user roles to ensure correct access controls.
              Monitoring and Alerting	Implements monitoring tools to track Kafka performance, health, and availability.	Detects issues early and allows proactive response.	Can add overhead to system if too frequent.	Test various failure scenarios to ensure proper alerting.
              Backups and Disaster Recovery	Configures backup systems and disaster recovery processes for Kafka.	Ensures data safety and quick recovery in case of major failure.	Backup and restore operations can be resource-intensive.	Test backup and restore processes to ensure data integrity.



Database	Database Replication	Maintains redundant database copies for high availability.	Ensure database availability and disaster recovery.	Potential data consistency issues.	Test failover scenarios and data consistency.
           Read Replicas	Uses read-only replicas to distribute read load and increase availability.	Improve read scalability and reduce primary database load.	Added latency due to replication lag.	Measure replication lag and ensure read consistency.

Load Balancers	Load Balancer Redundancy	Ensures multiple load balancers to distribute traffic and avoid single points of failure.	Provide load balancer redundancy and traffic distribution.	Additional infrastructure cost.	Simulate load balancer failure and ensure traffic distribution. 
                Global Load Balancing	Uses global load balancing to distribute traffic across different regions or data centers.	Ensure high availability and global failover capability.	More complex setup and additional latency.	Simulate region failure and verify traffic rerouting.

Storage	Data Backup	Regularly backs up data to prevent data loss.	Ensure data safety and recoverability.	Requires storage space and can affect performance.	Test data backup and restore process to ensure data integrity.                
                    Data Snapshotting	Takes snapshots of storage for quick recovery and rollback.	Allow quick recovery from data corruption or errors.	Can increase storage requirements.	Validate snapshot creation and test recovery scenarios.

Disaster Recovery	Multi-Region Deployment	Deploys infrastructure across multiple geographic regions for disaster recovery.	Ensure business continuity during major failures.	Higher infrastructure costs and added complexity.	Simulate region failure to ensure failover to backup region.

High Availability	Active-Active Configuration	Keeps multiple instances active to ensure high availability.	Ensure uninterrupted service during failures.	Requires consistent data across all active instances.	Simulate failure in one instance to ensure failover and service continuity.

RPO/RTO SLAs	Continuous Data Replication	Continuously replicates data to meet RPO/RTO SLAs.	Reduce potential data loss and improve recovery time.	Increased network traffic and storage requirements.	Test continuous replication and validate recovery within SLA.



