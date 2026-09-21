# Cloud Run one-click deployment script
param(
    [string]$ProjectId = "",
    [string]$Region = "us-central1"
)

if (-not $ProjectId) {
    $ProjectId = gcloud config get-value project 2>$null
}

if (-not $ProjectId) {
    Write-Error "Please specify your GCP Project ID: .\deploy.ps1 -ProjectId <YOUR_PROJECT_ID>"
    exit 1
}

Write-Host "==> Deploying GraphRank to Google Cloud Run" -ForegroundColor Cyan
Write-Host "Project ID: $ProjectId"
Write-Host "Region:     $Region"

# Build and deploy via gcloud builds
gcloud builds submit --config=graphrank/cloudbuild.yaml --substitutions=_REGION=$Region .

Write-Host "==> Deployment Complete! Visit your Cloud Run URL in the Google Cloud Console." -ForegroundColor Green
