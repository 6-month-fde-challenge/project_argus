# Project ARGUS documents

The client documents and the delivery team's write-ups are kept in a separate repo:
**[6-month-fde-challenge/argus](https://github.com/6-month-fde-challenge/argus)**

## Client documents

[`client_docs/`](https://github.com/6-month-fde-challenge/argus/tree/main/client_docs)

| # | Document |
|---|---|
| 01 | [Client Requirement Document (CRD)](https://github.com/6-month-fde-challenge/argus/blob/main/client_docs/01_Project_ARGUS_Client_Requirement_Document.docx) |
| 02 | [Architecture Document](https://github.com/6-month-fde-challenge/argus/blob/main/client_docs/02_Project_ARGUS_Architecture_Document.docx) |
| 03 | [Capacity Planning Document](https://github.com/6-month-fde-challenge/argus/blob/main/client_docs/03_Project_ARGUS_Capacity_Planning_Document.docx) |
| 04 | [High-Level Design (HLD)](https://github.com/6-month-fde-challenge/argus/blob/main/client_docs/04_Project_ARGUS_High-Level_Design_HLD_.docx) |
| 05 | [Low-Level Design (LLD)](https://github.com/6-month-fde-challenge/argus/blob/main/client_docs/05_Project_ARGUS_Low-Level_Design_LLD_.docx) |

## Day-wise write-ups

[`day_wise/`](https://github.com/6-month-fde-challenge/argus/tree/main/day_wise)

| Part | Topic |
|---|---|
| [Part 1](https://github.com/6-month-fde-challenge/argus/tree/main/day_wise/part_01) | Understanding the client |
| [Part 2](https://github.com/6-month-fde-challenge/argus/tree/main/day_wise/part_02) | Technical requirements |
| [Part 3](https://github.com/6-month-fde-challenge/argus/tree/main/day_wise/part_03) | Proposed solution, with [diagrams](https://github.com/6-month-fde-challenge/argus/tree/main/day_wise/part_03/diagrams) |
| [Part 4](https://github.com/6-month-fde-challenge/argus/tree/main/day_wise/part_04) | Client questions |
| [Part 5](https://github.com/6-month-fde-challenge/argus/tree/main/day_wise/part_05) | Mock meeting |

## Running the app locally

The app's Docs page reads these files from disk. `docker-compose.yml` mounts them from the folder one level up, so
clone the two repos like this:

```
01_argus/            <- github.com/6-month-fde-challenge/argus
  argus_solution/    <- github.com/6-month-fde-challenge/project_argus (this repo)
```
