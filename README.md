# stock-rl
This repo contains our project for CPE 800

# About

# Steps for Replication

 - <details>
    <summary style="font-size:larger;">Step 1: Set up AWS Infrastructure</summary>
    <br>
    The infrastructure for this project is contained in the <strong><font color=#0fb503>main.tf</font></strong> file. You can change this file to fit your specific architetcture needs but, if you just want to deploy the project for yourself there are some changes you will have to make. Those changes are as follows:
    <br>
    &nbsp;&nbsp;&nbsp;&nbsp;

    - Uncomment lines 90 - 97.
    <br>
    &nbsp;&nbsp;&nbsp;&nbsp;

    - Add your IP address to line 96
    <br>
    &nbsp;&nbsp;&nbsp;&nbsp;
    
    - Create your own key pair in AWS and replace the   
      keypair name on line 139 with the name of the 
      keypair you just created
    <br>
    &nbsp;&nbsp;&nbsp;&nbsp;

    - Add your email address to line 168

    </details>

<br>

 - <details>
    <summary style="font-size:larger;">Step 2: Set up CICD Pipeline</summary>
    <br>
    For this project a CICD pipeline is set up to automatically deploy and teardown the AWS infrastructure based on when the stock market opens and closes. Below are the steps to set up the CICD pipeline:
    <br>
    &nbsp;&nbsp;&nbsp;&nbsp;

    - You can follow the steps outlined [here](https://learn.hashicorp.com/tutorials/terraform/github-actions?in=terraform/automation) and tailor it to your needs. 
 
      
    </details>
