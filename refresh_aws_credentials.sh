#!/bin/bash

# Helper script to refresh AWS SSO credentials for the spyndex project
# Usage: ./refresh_aws_credentials.sh

echo "🔄 Refreshing AWS SSO credentials..."

# Login to AWS SSO
aws sso login --profile engineering

if [ $? -eq 0 ]; then
    echo "✅ AWS SSO login successful"
    echo "🔧 Environment is ready for spyndex S3 operations"
    echo ""
    echo "You can now run the notebook cells or use the spyndex S3 functionality"
    echo ""
    echo "📍 Key information:"
    echo "  • AWS Profile: engineering"
    echo "  • S3 Bucket: zulu-data-science" 
    echo "  • Data Prefix: bluesky-data/invermark/rgb/raw"
else
    echo "❌ AWS SSO login failed"
    echo "Please check your AWS configuration and try again"
    exit 1
fi 