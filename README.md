# Improving Network Intrusion Detection Systems

## Motivation

- Intrusion Detection Systems (IDSs) and Intrusion Prevention Systems (IPSs) are crucial for defending against network attacks.

- Unlike the commonly-used [KDDCUP99 outlier detection dataset](https://odds.cs.stonybrook.edu/smtp-kddcup99-dataset/), this dataset contains both benign and updated attack data, mimicking real-world scenarios, with an emphasis on generating realistic background traffic using the B-Profile system.
  
- It includes network traffic analysis results with labeled flows and feature definitions.
- It profiles the behavior of 25 users across various protocols like HTTP, HTTPS, FTP, SSH, and email.

### Key Dataset Features

The data was captured from July 3 to July 7, 2017, including benign traffic and various attacks (e.g., Brute Force, DoS, Heartbleed, Web Attack, Infiltration, Botnet, DDoS).

1. **Complete Network Configuration**: Includes diverse OS and network devices.
2. **Complete Traffic**: Involves a user profiling agent and different machines for victim and attack networks.
3. **Labelled Dataset**: Detailed labels for benign and attack data.
4. **Complete Interaction**: Covers internal LAN and internet communications.
5. **Complete Capture**: Utilizes mirror port for full traffic capture.
6. **Available Protocols**: Includes common protocols like HTTP, HTTPS, FTP, SSH, and email.
7. **Attack Diversity**: Features common attacks as per the 2016 McAfee report.
8. **Heterogeneity**: Traffic captured from the main Switch, including memory dumps and system calls.
9. **Feature Set**: Over 80 network flow features extracted.
10. **MetaData**: Comprehensive dataset description including time, attacks, flows, and labels.

## References

### Dataset

- [Liu, L., Engelen, G., Lynar, T., Essam, D., & Joosen, W. (2022). Error Prevalence in NIDS datasets: A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018. In Proceedings of the 2022 IEEE Conference. IEEE.](https://ieeexplore.ieee.org/abstract/document/9947235)

- [Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization. In Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP 2018) (pp. 108-116).](https://www.scitepress.org/papers/2018/66398/66398.pdf)

### Theory

### MLOps

- [On how to use/interpret Dask's diagnosis dashboard](https://docs.dask.org/en/stable/dashboard.html)
