from setuptools import setup, find_packages

# Function to read the requirements from requirements.txt
def read_requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()

setup(
    name='rec-sys-model',
    version='0.1.0',
    packages=find_packages(),
    # 💡 Use the function to populate install_requires
    install_requires=read_requirements(),
    # Include other necessary files for the package
    include_package_data=True,
    zip_safe=False,
)
