terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

# Configure the AWS Provider
provider "aws" {
  region = "us-east-1"
}

# Create a VPC
resource "aws_vpc" "stock-rl-vpc" {
  cidr_block = "10.0.0.0/16"
}

# Create an Internet Gateway
resource "aws_internet_gateway" "stock-rl-igw" {
    vpc_id = aws_vpc.stock-rl-vpc.id

    tags = {
        Name = "stock-rl-igw"
    }
}

# Create an Internet Gateway Attachment
resource "aws_internet_gateway_attachment" "stock-rl-igw-attachment" {
    vpc_id = aws_vpc.stock-rl-vpc.id
    internet_gateway_id = aws_internet_gateway.stock-rl-igw.id
}

# Create a Public Subnet
resource "aws_subnet" "stock-rl-public-subnet" {
    vpc_id = aws_vpc.stock-rl-vpc.id
    cidr_block = "10.0.128.0/17"
    availability_zone = "us-east-1a"


    tags = {
        Name = "stock-rl-public-subnet"
    }
}

# Create a Route Table 
resource "aws_route_table" "stock-rl-public-route-table" {
    vpc_id = aws_vpc.stock-rl-vpc.id

    route {
        cidr_block = "0.0.0.0/0"
        gateway_id = aws_internet_gateway.stock-rl-igw.id
    }

    tags = {
        Name = "stock-rl-public-route-table"
    }
}

# Create a Route Table Association
resource "aws_route_table_association" "stock-rl-public-route-table-association" {
    subnet_id = aws_subnet.stock-rl-public-subnet.id
    route_table_id = aws_route_table.stock-rl-public-route-table.id
}

# Create a Security Group
resource "aws_security_group" "stock-rl-security-group" {
    name = "stock-rl-security-group"
    description = "Allow inbound traffic from the Internet"
    vpc_id = aws_vpc.stock-rl-vpc.id

    ingress {
        description = "Only Allow HTTPS from the Internet"
        from_port = 443
        to_port = 443
        protocol = "tcp"
    }

    egress = {
        description = "Allow all outbound traffic"
        from_port = 0
        to_port = 0
        protocol = "-1"
    }
}
# Create an EC2 Instance
# will wait to see what we need for sure before setting this up
