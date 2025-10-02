/**
 * Terraform Outputs for Deep Research AI Agent
 *
 * Export important resource information for use by other systems.
 */

# ==================== Cloud Run Outputs ====================

output "cloud_run_url" {
  description = "URL of the deployed Cloud Run service"
  value       = google_cloud_run_service.research_agent.status[0].url
}

output "cloud_run_service_name" {
  description = "Cloud Run service name"
  value       = google_cloud_run_service.research_agent.name
}

# ==================== Database Outputs ====================

output "database_instance_name" {
  description = "Cloud SQL instance name"
  value       = google_sql_database_instance.research_db.name
}

output "database_connection_name" {
  description = "Cloud SQL connection name for connecting via Unix socket"
  value       = google_sql_database_instance.research_db.connection_name
}

output "database_ip_address" {
  description = "Public IP address of the Cloud SQL instance"
  value       = google_sql_database_instance.research_db.ip_address[0].ip_address
}

output "database_name" {
  description = "Database name"
  value       = google_sql_database.research_database.name
}

output "database_user" {
  description = "Database user"
  value       = google_sql_user.research_user.name
}

# ==================== Storage Outputs ====================

output "storage_bucket_name" {
  description = "Cloud Storage bucket name for reports and logs"
  value       = google_storage_bucket.reports_bucket.name
}

output "storage_bucket_url" {
  description = "Cloud Storage bucket URL"
  value       = google_storage_bucket.reports_bucket.url
}

# ==================== Service Account Outputs ====================

output "service_account_email" {
  description = "Service account email for the research agent"
  value       = google_service_account.research_agent_sa.email
}

output "service_account_id" {
  description = "Service account ID"
  value       = google_service_account.research_agent_sa.account_id
}

# ==================== Secret Manager Outputs ====================

output "anthropic_secret_id" {
  description = "Secret Manager ID for Anthropic API key"
  value       = google_secret_manager_secret.anthropic_api_key.secret_id
}

output "google_ai_secret_id" {
  description = "Secret Manager ID for Google AI API key"
  value       = google_secret_manager_secret.google_api_key.secret_id
}

output "serpapi_secret_id" {
  description = "Secret Manager ID for SerpApi key"
  value       = google_secret_manager_secret.serpapi_key.secret_id
}

output "db_password_secret_id" {
  description = "Secret Manager ID for database password"
  value       = google_secret_manager_secret.db_password.secret_id
}

# ==================== Connection Information ====================

output "connection_info" {
  description = "Summary of connection information"
  value = {
    cloud_run_url          = google_cloud_run_service.research_agent.status[0].url
    database_connection    = google_sql_database_instance.research_db.connection_name
    storage_bucket         = google_storage_bucket.reports_bucket.name
    service_account        = google_service_account.research_agent_sa.email
  }
}

# ==================== Deployment Instructions ====================

output "deployment_instructions" {
  description = "Instructions for deploying the application"
  value = <<-EOT

  Deployment Instructions:
  =======================

  1. Build and push Docker image:
     docker build -t gcr.io/${var.gcp_project_id}/research-agent:latest .
     docker push gcr.io/${var.gcp_project_id}/research-agent:latest

  2. Deploy to Cloud Run:
     gcloud run deploy research-agent \
       --image gcr.io/${var.gcp_project_id}/research-agent:latest \
       --platform managed \
       --region ${var.gcp_region} \
       --allow-unauthenticated

  3. Access the application:
     ${google_cloud_run_service.research_agent.status[0].url}

  4. Connect to database:
     gcloud sql connect ${google_sql_database_instance.research_db.name} \
       --user=${google_sql_user.research_user.name} \
       --database=${google_sql_database.research_database.name}

  5. View logs:
     gcloud run services logs read research-agent --region ${var.gcp_region}

  EOT
}

# ==================== Cost Estimation ====================

output "estimated_monthly_cost_usd" {
  description = "Rough estimate of monthly costs (for reference only)"
  value = <<-EOT

  Estimated Monthly Costs (approximate):
  =====================================

  Cloud SQL (db-f1-micro):    ~$7-10/month
  Cloud Run (minimal usage):  ~$0-5/month (pay per request)
  Cloud Storage:              ~$0.026/GB/month
  Secret Manager:             ~$0.06 per 10K accesses

  Total Estimated:            ~$10-20/month for light usage

  Note: Actual costs depend on usage patterns, data storage, and API calls.
  Monitor costs at: https://console.cloud.google.com/billing

  EOT
}
