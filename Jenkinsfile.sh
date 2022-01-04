set -e
echo "----------------------------------------------"
echo "$JOB_NAME build $BUILD_NUMBER"
WORKSPACE_ROOT="$WORKSPACE/$BUILD_NUMBER"
echo "Setting up build in $WORKSPACE_ROOT"
PROJECTS_DIR_REL="projects"
PROJECTS_DIR=$WORKSPACE_ROOT/$PROJECTS_DIR_REL
echo "----------------------------------------------"
echo "\nEnvironment:"
df -h .
echo "----------------------------------------------"
cat /proc/meminfo
echo "----------------------------------------------"

#build using docker file in directory:
cd $PROJECTS_DIR/montblanc
IMAGENAME="mb_py38"
docker build -t "$IMAGENAME:$BUILD_NUMBER" --no-cache=false -f .ci/py3.8.docker .
IMAGENAME="mb_py36"
docker build -t "$IMAGENAME:$BUILD_NUMBER" --no-cache=false -f .ci/py3.6.docker .
IMAGENAME="mb_py38nonvss"
docker build -t "$IMAGENAME:$BUILD_NUMBER" --no-cache=false -f .ci/withoutnvcc.py3.8.docker .