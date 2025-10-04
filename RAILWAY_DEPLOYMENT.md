# Railway Deployment Guide for Exoplings

This guide will help you deploy your Exoplings Flask application to Railway.com.

## Prerequisites

1. **Railway Account**: Sign up at [railway.app](https://railway.app)
2. **GitHub Repository**: Ensure your code is pushed to GitHub
3. **Railway CLI** (optional): Install from [railway.app/cli](https://docs.railway.app/develop/cli)

## Deployment Steps

### Method 1: GitHub Integration (Recommended)

1. **Connect Railway to GitHub:**
   - Go to [railway.app](https://railway.app)
   - Click "Start a New Project"
   - Select "Deploy from GitHub repo"
   - Authorize Railway to access your GitHub account
   - Select your `spaceapps-2025` repository

2. **Configure Environment Variables:**
   - In your Railway project dashboard, go to the "Variables" tab
   - Add the following environment variables:
     ```
     SECRET_KEY=your-super-secret-key-here-generate-a-strong-one
     FLASK_ENV=production
     PYTHONPATH=src
     PORT=8080
     ```
   - **Important**: Generate a strong SECRET_KEY using Python:
     ```python
     import secrets
     print(secrets.token_hex(32))
     ```

3. **Deploy:**
   - Railway will automatically detect your `railway.toml` and `Procfile`
   - The deployment will start automatically
   - Monitor the build logs in the Railway dashboard

### Method 2: Railway CLI

1. **Install Railway CLI:**
   ```bash
   npm install -g @railway/cli
   ```

2. **Login to Railway:**
   ```bash
   railway login
   ```

3. **Initialize and Deploy:**
   ```bash
   cd /path/to/your/spaceapps-2025
   railway init
   railway up
   ```

4. **Set Environment Variables:**
   ```bash
   railway variables set SECRET_KEY=your-super-secret-key-here
   railway variables set FLASK_ENV=production
   railway variables set PYTHONPATH=src
   ```

## Configuration Files Created

The following files have been created for Railway deployment:

- **`railway.toml`**: Railway-specific configuration
- **`Procfile`**: Process definition for web server
- **`requirements.txt`**: Python dependencies (generated from pyproject.toml)
- **`.env.example`**: Example environment variables
- **`.railwayignore`**: Files to exclude from deployment

## Important Notes

### File Uploads
- Railway provides ephemeral storage, so uploaded files will be lost on container restarts
- Consider using a cloud storage service (AWS S3, Google Cloud Storage) for persistent file storage
- The current upload folder is configurable via the `UPLOAD_FOLDER` environment variable

### AI Models
- Your AI model files (`CNN_1D.pth`, `Inferrer_Ultra.pth`) are included in the deployment
- Make sure these files are not too large (Railway has size limits)
- Consider using model compression or external model storage if needed

### Static Files
- Static files (CSS, JS, images) are served by Flask in production
- The background image and other assets should work correctly

## Troubleshooting

### Common Issues:

1. **Build Fails:**
   - Check that all dependencies in `pyproject.toml` are available
   - Verify Python version compatibility (>=3.10)

2. **App Doesn't Start:**
   - Check the logs in Railway dashboard
   - Ensure `PYTHONPATH=src` is set as environment variable
   - Verify the start command in `railway.toml`

3. **Static Files Not Loading:**
   - Check that static files are properly committed to Git
   - Verify Flask static folder configuration

4. **Large Deployment Size:**
   - Use `.railwayignore` to exclude unnecessary files
   - Consider compressing AI model files

### Getting Logs:
```bash
railway logs
```

### Accessing Your App:
After successful deployment, Railway will provide a URL like:
`https://your-app-name.up.railway.app`

## Production Checklist

- [ ] Set a strong SECRET_KEY
- [ ] Set FLASK_ENV=production
- [ ] Configure PYTHONPATH=src
- [ ] Test file upload functionality
- [ ] Verify AI model inference works
- [ ] Test all routes and pages
- [ ] Monitor application logs

## Support

If you encounter issues:

1. Check Railway documentation: [docs.railway.app](https://docs.railway.app)
2. Review application logs in Railway dashboard
3. Ensure all environment variables are properly set
4. Verify that your GitHub repository is up to date

## Cost Considerations

Railway offers:
- **Free tier**: $5/month of usage credits
- **Pro tier**: $20/month for higher usage

Monitor your usage in the Railway dashboard to avoid unexpected charges.