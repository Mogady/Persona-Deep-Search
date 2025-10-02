/**
 * Deep Research AI Agent - GCP Infrastructure
 *
 * Terraform configuration for deploying the research agent on Google Cloud Platform.
 * This configuration creates minimal resources for cost-effective testing and deployment.
 */

terraform {
  required_version = ">= 1.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  # Uncomment to use remote backend
  # backend "gcs" {
  #   bucket = "research-agent-terraform-state"
  #   prefix = "terraform/state"
  # }
}

# ==================== Provider Configuration ====================

provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

# ==================== Enable Required APIs ====================

resource "google_project_service" "cloud_sql_admin" {
  service            = "sqladmin.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "secret_manager" {
  service            = "secretmanager.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloud_run" {
  service            = "run.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloud_storage" {
  service            = "storage.googleapis.com"
  disable_on_destroy = false
}

# ==================== Cloud SQL PostgreSQL Database ====================

resource "google_sql_database_instance" "research_db" {
  name             = var.db_instance_name
  database_version = "POSTGRES_15"
  region           = var.gcp_region

  # Use minimal machine type for cost efficiency
  settings {
    tier              = var.db_tier
    availability_type = "ZONAL" # Use REGIONAL for production
    disk_size         = 10       # GB
    disk_type         = "PD_SSD"

    backup_configuration {
      enabled            = var.enable_backups
      start_time         = "03:00"
      point_in_time_recovery_enabled = false
      transaction_log_retention_days = 7
      backup_retention_settings {
        retained_backups = 7
      }
    }

    maintenance_window {
      day          = 7 # Sunday
      hour         = 3
      update_track = "stable"
    }

    ip_configuration {
      ipv4_enabled    = true
      require_ssl     = true
      private_network = var.vpc_network_id
    }

    database_flags {
      name  = "max_connections"
      value = "100"
    }
  }

  deletion_protection = var.enable_deletion_protection

  depends_on = [google_project_service.cloud_sql_admin]
}

resource "google_sql_database" "research_database" {
  name     = var.db_name
  instance = google_sql_database_instance.research_db.name
}

resource "google_sql_user" "research_user" {
  name     = var.db_user
  instance = google_sql_database_instance.research_db.name
  password = var.db_password
}

# ==================== Secret Manager ====================

resource "google_secret_manager_secret" "anthropic_api_key" {
  secret_id = "anthropic-api-key"

  replication {
    auto {}
  }

  depends_on = [google_project_service.secret_manager]
}

resource "google_secret_manager_secret_version" "anthropic_api_key_version" {
  secret      = google_secret_manager_secret.anthropic_api_key.id
  secret_data = var.anthropic_api_key
}

resource "google_secret_manager_secret" "google_api_key" {
  secret_id = "google-ai-api-key"

  replication {
    auto {}
  }

  depends_on = [google_project_service.secret_manager]
}

resource "google_secret_manager_secret_version" "google_api_key_version" {
  secret      = google_secret_manager_secret.google_api_key.id
  secret_data = var.google_api_key
}

resource "google_secret_manager_secret" "serpapi_key" {
  secret_id = "serpapi-key"

  replication {
    auto {}
  }

  depends_on = [google_project_service.secret_manager]
}

resource "google_secret_manager_secret_version" "serpapi_key_version" {
  secret      = google_secret_manager_secret.serpapi_key.id
  secret_data = var.serpapi_key
}

resource "google_secret_manager_secret" "db_password" {
  secret_id = "db-password"

  replication {
    auto {}
  }

  depends_on = [google_project_service.secret_manager]
}

resource "google_secret_manager_secret_version" "db_password_version" {
  secret      = google_secret_manager_secret.db_password.id
  secret_data = var.db_password
}

# ==================== Cloud Storage Bucket ====================

resource "google_storage_bucket" "reports_bucket" {
  name     = var.storage_bucket_name
  location = var.gcp_region

  uniform_bucket_level_access = true

  versioning {
    enabled = false
  }

  lifecycle_rule {
    condition {
      age = var.data_retention_days
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [google_project_service.cloud_storage]
}

# ==================== Service Account ====================

resource "google_service_account" "research_agent_sa" {
  account_id   = "research-agent"
  display_name = "Research Agent Service Account"
  description  = "Service account for Deep Research AI Agent"
}

# IAM permissions for service account
resource "google_project_iam_member" "sa_cloud_sql_client" {
  project = var.gcp_project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.research_agent_sa.email}"
}

resource "google_project_iam_member" "sa_secret_accessor" {
  project = var.gcp_project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.research_agent_sa.email}"
}

resource "google_storage_bucket_iam_member" "sa_storage_admin" {
  bucket = google_storage_bucket.reports_bucket.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.research_agent_sa.email}"
}

# ==================== Cloud Run Service ====================

resource "google_cloud_run_service" "research_agent" {
  name     = var.cloud_run_service_name
  location = var.gcp_region

  template {
    spec {
      service_account_name = google_service_account.research_agent_sa.email

      containers {
        image = var.container_image

        ports {
          container_port = 8000
        }

        resources {
          limits = {
            cpu    = var.cloud_run_cpu
            memory = var.cloud_run_memory
          }
        }

        env {
          name  = "GCP_PROJECT_ID"
          value = var.gcp_project_id
        }

        env {
          name  = "ENVIRONMENT"
          value = var.environment
        }

        env {
          name  = "USE_SECRET_MANAGER"
          value = "true"
        }

        env {
          name  = "CLOUD_SQL_CONNECTION_NAME"
          value = google_sql_database_instance.research_db.connection_name
        }

        env {
          name  = "DB_NAME"
          value = var.db_name
        }

        env {
          name  = "DB_USER"
          value = var.db_user
        }

        env {
          name  = "GCP_STORAGE_BUCKET"
          value = google_storage_bucket.reports_bucket.name
        }

        # Secret references (mounted from Secret Manager)
        env {
          name = "ANTHROPIC_API_KEY"
          value_from {
            secret_key_ref {
              name = google_secret_manager_secret.anthropic_api_key.secret_id
              key  = "latest"
            }
          }
        }

        env {
          name = "GOOGLE_API_KEY"
          value_from {
            secret_key_ref {
              name = google_secret_manager_secret.google_api_key.secret_id
              key  = "latest"
            }
          }
        }

        env {
          name = "SERPAPI_KEY"
          value_from {
            secret_key_ref {
              name = google_secret_manager_secret.serpapi_key.secret_id
              key  = "latest"
            }
          }
        }

        env {
          name = "DB_PASSWORD"
          value_from {
            secret_key_ref {
              name = google_secret_manager_secret.db_password.secret_id
              key  = "latest"
            }
          }
        }
      }

      # Cloud SQL connection using Unix socket
      # Uncomment if using Cloud SQL Proxy sidecar
      # containers {
      #   image = "gcr.io/cloudsql-docker/gce-proxy:latest"
      #   command = [
      #     "/cloud_sql_proxy",
      #     "-instances=${google_sql_database_instance.research_db.connection_name}=tcp:5432"
      #   ]
      # }
    }

    metadata {
      annotations = {
        "autoscaling.knative.dev/maxScale"      = var.cloud_run_max_instances
        "autoscaling.knative.dev/minScale"      = var.cloud_run_min_instances
        "run.googleapis.com/cloudsql-instances" = google_sql_database_instance.research_db.connection_name
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  depends_on = [
    google_project_service.cloud_run,
    google_sql_database_instance.research_db
  ]
}

# Make Cloud Run service public (remove for authentication)
resource "google_cloud_run_service_iam_member" "public_access" {
  count = var.allow_public_access ? 1 : 0

  service  = google_cloud_run_service.research_agent.name
  location = google_cloud_run_service.research_agent.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ==================== Outputs ====================

output "cloud_run_url" {
  description = "URL of the deployed Cloud Run service"
  value       = google_cloud_run_service.research_agent.status[0].url
}

output "database_connection_name" {
  description = "Cloud SQL connection name"
  value       = google_sql_database_instance.research_db.connection_name
}

output "database_ip" {
  description = "Database IP address"
  value       = google_sql_database_instance.research_db.ip_address[0].ip_address
}

output "storage_bucket" {
  description = "Cloud Storage bucket for reports"
  value       = google_storage_bucket.reports_bucket.name
}

output "service_account_email" {
  description = "Service account email"
  value       = google_service_account.research_agent_sa.email
}
