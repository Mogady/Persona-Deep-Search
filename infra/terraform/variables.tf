/**
 * Terraform Variables for Deep Research AI Agent
 *
 * Define all configurable parameters for the infrastructure.
 */

# ==================== GCP Configuration ====================

variable "gcp_project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "gcp_region" {
  description = "GCP Region for resources"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment name (development, staging, production)"
  type        = string
  default     = "development"

  validation {
    condition     = contains(["development", "staging", "production"], var.environment)
    error_message = "Environment must be development, staging, or production."
  }
}

# ==================== Database Configuration ====================

variable "db_instance_name" {
  description = "Cloud SQL instance name"
  type        = string
  default     = "research-agent-db"
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "research_agent"
}

variable "db_user" {
  description = "Database user"
  type        = string
  default     = "research_user"
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

variable "db_tier" {
  description = "Cloud SQL machine tier"
  type        = string
  default     = "db-f1-micro" # Minimal for testing, use db-custom-2-4096 for production

  validation {
    condition     = can(regex("^db-", var.db_tier))
    error_message = "DB tier must start with 'db-'."
  }
}

variable "enable_backups" {
  description = "Enable automated database backups"
  type        = bool
  default     = false
}

variable "enable_deletion_protection" {
  description = "Enable deletion protection for database"
  type        = bool
  default     = false
}

variable "vpc_network_id" {
  description = "VPC network ID for private IP (optional)"
  type        = string
  default     = null
}

# ==================== API Keys (Secrets) ====================

variable "anthropic_api_key" {
  description = "Anthropic API key for Claude"
  type        = string
  sensitive   = true
}

variable "google_api_key" {
  description = "Google AI API key for Gemini"
  type        = string
  sensitive   = true
}

variable "serpapi_key" {
  description = "SerpApi key for search"
  type        = string
  sensitive   = true
}

# ==================== Storage Configuration ====================

variable "storage_bucket_name" {
  description = "Cloud Storage bucket name for reports and logs"
  type        = string
}

variable "data_retention_days" {
  description = "Number of days to retain data in storage"
  type        = number
  default     = 30

  validation {
    condition     = var.data_retention_days > 0 && var.data_retention_days <= 365
    error_message = "Data retention must be between 1 and 365 days."
  }
}

# ==================== Cloud Run Configuration ====================

variable "cloud_run_service_name" {
  description = "Cloud Run service name"
  type        = string
  default     = "research-agent"
}

variable "container_image" {
  description = "Container image to deploy"
  type        = string
  default     = "gcr.io/PROJECT_ID/research-agent:latest"
}

variable "cloud_run_cpu" {
  description = "CPU allocation for Cloud Run"
  type        = string
  default     = "1" # 1 vCPU

  validation {
    condition     = contains(["1", "2", "4"], var.cloud_run_cpu)
    error_message = "CPU must be 1, 2, or 4."
  }
}

variable "cloud_run_memory" {
  description = "Memory allocation for Cloud Run"
  type        = string
  default     = "512Mi" # 512 MB

  validation {
    condition     = can(regex("^[0-9]+(Mi|Gi)$", var.cloud_run_memory))
    error_message = "Memory must be in format like '512Mi' or '2Gi'."
  }
}

variable "cloud_run_min_instances" {
  description = "Minimum number of Cloud Run instances"
  type        = string
  default     = "0" # Scale to zero when idle
}

variable "cloud_run_max_instances" {
  description = "Maximum number of Cloud Run instances"
  type        = string
  default     = "10"
}

variable "allow_public_access" {
  description = "Allow public access to Cloud Run service"
  type        = bool
  default     = true
}

# ==================== Tags and Labels ====================

variable "common_labels" {
  description = "Common labels to apply to all resources"
  type        = map(string)
  default = {
    project     = "deep-research-agent"
    managed_by  = "terraform"
  }
}
