
Note for Microsoft AI MLOps

# Introduction
## AI 
* machine learning
* anomaly detection
* computer vision
* NLP
* knowledge mining
## Azure
* automatic ML
* Azure ML designer
* data/compute management
* pipelines

## Challenges, Risks
* bias
* errors
* data exposed
* solution not for everyone
* users trust a complex system
* liability for AI-driven decision

## Responsbile AI
* fairness -- reduce bias
* reliability and safety -- reduce errors
* privacy and security -- data exposed
* inclusiveness -- diversity
* transparency
* accountability

# Azure openAI service

## Introduction
four components
* pre-trained model
* customization capabilities
* detect and mitigate harmful use
* enterprise-grade security and access control
three groups
* ML platform
* Cognitive service
*  applied AI service
models
* GPT-4
* GPT-3.5
* DALL-E
* embedding models
Code
* codex
* copilot

# MLOps
to make machine learning lifecycle scalable:

  train -> package -> validate -> deploy -> monitor -> retrain
  
agile planning, source control, automation

## DevOps
goal of MLOps: creating, deploying and monitoring robust and reproducible models
* ML: EDA, feature engineering, model training and tuning
* DEV: plan, create, verify, package
* OPS: release, configure, monitor

### CI/CD
CI: create and verify, include 
* refactor exploratory code into scripts
* linting to ceck errors
* unit testing

CD: package, deploy to pre-production environment, production environment, source control 

Source control: git-based respository (Azure Repos, github repos)

### Agile planning
* isolate work into sprints 
* use Azure Boards or Github issues

###IaC (Infrastructure as code)

### Azure DevOps
* Boards
* Repos
* Pipelines: CI/CD

### Github
*  Issues
*  Repos
*  Actions: automatic workflow

## Azure ML Service
Role Based Access Control (RBAC): owner, contributor, reader

compute resources: instance, cluster, inference cluster, attached compute

datastores: workspacefilestore, workspaceblobstore

environment: Azure container registry

type of jobs:
* command
* sweep: hyperparameter tuning
* pipeline

## Design a data ingestion strtegy
six steps
* define problem
* get data
* prepare -- ELT, ELT
* train
* integrate: deploy to an endpoint
* monitor

Identify
* data source: CRM, SQL, IoT, ...
* data format


## Trigger Github Actions with feature-based dev
* data science team -- model development -- feature branch
* software team -- deploy

on pull_request; on push 

## linting and unit testing

## environment
* development
* staging
* production
## Deploy





