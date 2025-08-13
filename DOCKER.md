## Configuration

### Environment Variables
- `MODELS_DIR`: Path to model files
- `OUTPUT_DIR`: Path for metadata and generated files
- `UPLOAD_DIR`: Path for uploaded files
- `FLASK_APP`: Path to Flask application
- `FLASK_ENV`: `production` or `development`

### Volumes
- `/data/models`: Model files (read-only)
- `/data/output`: Metadata and generated files
- `/data/uploads`: Uploaded files

### Ports
- `5000`: Web interface

## Maintenance

### Updates

#### Docker Container
```bash
docker pull civitai-manager:latest
docker stop civitai-manager
docker rm civitai-manager
# Re-run docker run command
```

#### TrueNAS Scale App
1. Go to "Apps" -> "Installed Applications"
2. Find "Civitai Manager"
3. Click "Upgrade"

### Backups
Important directories to backup:
- `/data/output`: Contains all metadata and generated files
- `/data/uploads`: Contains uploaded models

### Logs
View logs:
```bash
# Docker
docker logs civitai-manager

# TrueNAS Scale App
kubectl logs -n ix-civitai-manager deployment/civitai-manager
```

## Troubleshooting

### Common Issues

#### Permission Errors
Check permissions:
```bash
ls -la /mnt/pool/models
ls -la /mnt/pool/civitai-output
ls -la /mnt/pool/civitai-uploads
```

Fix permissions:
```bash
chown -R 1000:1000 /mnt/pool/civitai-output
chown -R 1000:1000 /mnt/pool/civitai-uploads
chmod -R 755 /mnt/pool/models
```

#### Container Won't Start
1. Check logs:
   ```bash
   docker logs civitai-manager
   ```

2. Verify volume mounts:
   ```bash
   docker inspect civitai-manager
   ```

3. Ensure directories exist and have correct permissions

#### Web Interface Not Accessible
1. Check if container is running:
   ```bash
   docker ps | grep civitai-manager
   ```

2. Verify port mapping:
   ```bash
   docker port civitai-manager
   ```

3. Check TrueNAS Scale network settings
