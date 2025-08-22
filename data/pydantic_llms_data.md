========================
CODE SNIPPETS
========================
TITLE: Install Pydantic from GitHub repository (uv)
DESCRIPTION: Installs the Pydantic library directly from its main branch on GitHub using uv. This is useful for installing the latest development version.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/install.md#_snippet_9

LANGUAGE: bash
CODE:
```
uv add 'git+https://github.com/pydantic/pydantic@main'
```

----------------------------------------

TITLE: Install Pydantic from GitHub repository with extras (uv)
DESCRIPTION: Installs Pydantic from its GitHub repository with specified extra dependencies (e.g., 'email', 'timezone') using uv.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/install.md#_snippet_10

LANGUAGE: bash
CODE:
```
uv add 'git+https://github.com/pydantic/pydantic@main#egg=pydantic[email,timezone]'
```

----------------------------------------

TITLE: Install Pydantic from GitHub repository (pip)
DESCRIPTION: Installs the Pydantic library directly from its main branch on GitHub using pip. This is useful for installing the latest development version.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/install.md#_snippet_7

LANGUAGE: bash
CODE:
```
pip install 'git+https://github.com/pydantic/pydantic@main'
```

----------------------------------------

TITLE: Install Pydantic from GitHub repository with extras (pip)
DESCRIPTION: Installs Pydantic from its GitHub repository with specified extra dependencies (e.g., 'email', 'timezone') using pip.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/install.md#_snippet_8

LANGUAGE: bash
CODE:
```
pip install 'git+https://github.com/pydantic/pydantic@main#egg=pydantic[email,timezone]'
```

----------------------------------------

TITLE: Install Pydantic with optional email support (uv)
DESCRIPTION: Installs Pydantic with the 'email' extra dependency using uv, enabling email validation features.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/install.md#_snippet_5

LANGUAGE: bash
CODE:
```
uv add 'pydantic[email]'
```

----------------------------------------

TITLE: Install Pydantic with uv
DESCRIPTION: Installs the Pydantic library using the uv package manager. uv is a fast, modern Python package installer.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/install.md#_snippet_1

LANGUAGE: bash
CODE:
```
uv add pydantic
```

----------------------------------------

TITLE: Install Pydantic with optional email support (pip)
DESCRIPTION: Installs Pydantic with the 'email' extra dependency using pip, enabling email validation features.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/install.md#_snippet_3

LANGUAGE: bash
CODE:
```
pip install 'pydantic[email]'
```

----------------------------------------

TITLE: Install Pydantic with pip
DESCRIPTION: Installs the Pydantic library using the pip package manager. This is the standard method for installing Python packages.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/install.md#_snippet_0

LANGUAGE: bash
CODE:
```
pip install pydantic
```

----------------------------------------

TITLE: Install Pydantic with optional email and timezone support (uv)
DESCRIPTION: Installs Pydantic with both 'email' and 'timezone' extra dependencies using uv, providing full support for email validation and timezone handling.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/install.md#_snippet_6

LANGUAGE: bash
CODE:
```
uv add 'pydantic[email,timezone]'
```

----------------------------------------

TITLE: Install Pydantic and Dependencies
DESCRIPTION: This command installs Pydantic, its dependencies, test dependencies, and documentation dependencies using the project's Makefile. It sets up the complete development environment.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/contributing.md#_snippet_4

LANGUAGE: bash
CODE:
```
make install
```

----------------------------------------

TITLE: Install Pydantic with optional email and timezone support (pip)
DESCRIPTION: Installs Pydantic with both 'email' and 'timezone' extra dependencies using pip, providing full support for email validation and timezone handling.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/install.md#_snippet_4

LANGUAGE: bash
CODE:
```
pip install 'pydantic[email,timezone]'
```

----------------------------------------

TITLE: Update Documentation Examples with Pytest
DESCRIPTION: Command to run Pydantic's documentation tests and automatically update any outdated code examples found within the documentation files. This ensures examples remain accurate and runnable.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/contributing.md#_snippet_11

LANGUAGE: bash
CODE:
```
# Run tests and update code examples
pytest tests/test_docs.py --update-examples
```

----------------------------------------

TITLE: Install Development Tools
DESCRIPTION: Commands to install essential development tools like uv (a Python package installer and virtual environment manager) and pre-commit (a framework for managing and automating pre-commit hooks).

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/contributing.md#_snippet_3

LANGUAGE: bash
CODE:
```
pipx install uv
pipx install pre-commit
```

----------------------------------------

TITLE: Install Bump Pydantic Tool
DESCRIPTION: Installs the 'bump-pydantic' tool, a beta utility designed to help automate code transformations for Pydantic V1 to V2 migration. It's installed via pip.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_1

LANGUAGE: bash
CODE:
```
pip install bump-pydantic
```

----------------------------------------

TITLE: Install Pydantic with conda
DESCRIPTION: Installs the Pydantic library from the conda-forge channel using the conda package manager. This method is suitable for users within the Anaconda ecosystem.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/install.md#_snippet_2

LANGUAGE: bash
CODE:
```
conda install pydantic -c conda-forge
```

----------------------------------------

TITLE: Configure Mypy with Pydantic Plugin (pyproject.toml)
DESCRIPTION: Example configuration for pyproject.toml to enable the pydantic plugin and set various mypy and pydantic-mypy specific strictness flags. This setup enhances type checking for Pydantic models.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/integrations/mypy.md#_snippet_3

LANGUAGE: toml
CODE:
```
[tool.mypy]
plugins = ["pydantic.mypy"]

follow_imports = "silent"
warn_redundant_casts = true
warn_unused_ignores = true
disallow_any_generics = true
no_implicit_reexport = true
disallow_untyped_defs = true

[tool.pydantic-mypy]
init_forbid_extra = true
init_typed = true
warn_required_dynamic_aliases = true
```

----------------------------------------

TITLE: Install datamodel-code-generator
DESCRIPTION: Installs the datamodel-code-generator library using pip. This is the initial step required to utilize the tool for generating Pydantic models from data schemas.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/integrations/datamodel_code_generator.md#_snippet_0

LANGUAGE: bash
CODE:
```
pip install datamodel-code-generator
```

----------------------------------------

TITLE: Configure Mypy with Pydantic Plugin (mypy.ini)
DESCRIPTION: Example configuration for mypy.ini to enable the pydantic plugin and set various mypy and pydantic-mypy specific strictness flags. This setup enhances type checking for Pydantic models.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/integrations/mypy.md#_snippet_2

LANGUAGE: ini
CODE:
```
[mypy]
plugins = pydantic.mypy

follow_imports = silent
warn_redundant_casts = True
warn_unused_ignores = True
disallow_any_generics = True
no_implicit_reexport = True
disallow_untyped_defs = True

[pydantic-mypy]
init_forbid_extra = True
init_typed = True
warn_required_dynamic_aliases = True
```

----------------------------------------

TITLE: Install Pydantic V2
DESCRIPTION: Installs the latest production release of Pydantic V2 using pip. This command ensures you have the most up-to-date version for new projects or upgrades.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_0

LANGUAGE: bash
CODE:
```
pip install -U pydantic
```

----------------------------------------

TITLE: Install Pydantic V1
DESCRIPTION: Installs a specific version of Pydantic V1 using pip. This is useful if you need to maintain compatibility with Pydantic V1 for existing projects or specific features.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_3

LANGUAGE: bash
CODE:
```
pip install "pydantic==1.*
"
```

----------------------------------------

TITLE: Pydantic with ARQ Job Queue Example
DESCRIPTION: Demonstrates defining a Pydantic model for job data, serializing it for enqueueing, and validating/deserializing it during job processing with ARQ. Requires Redis and ARQ installed.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/examples/queues.md#_snippet_3

LANGUAGE: python
CODE:
```
import asyncio
from typing import Any

from arq import create_pool
from arq.connections import RedisSettings

from pydantic import BaseModel, EmailStr


class User(BaseModel):
    id: int
    name: str
    email: EmailStr


REDIS_SETTINGS = RedisSettings()


async def process_user(ctx: dict[str, Any], user_data: dict[str, Any]) -> None:
    user = User.model_validate(user_data)
    print(f'Processing user: {repr(user)}')


async def enqueue_jobs(redis):
    user1 = User(id=1, name='John Doe', email='john@example.com')
    user2 = User(id=2, name='Jane Doe', email='jane@example.com')

    await redis.enqueue_job('process_user', user1.model_dump())
    print(f'Enqueued user: {repr(user1)}')

    await redis.enqueue_job('process_user', user2.model_dump())
    print(f'Enqueued user: {repr(user2)}')


class WorkerSettings:
    functions = [process_user]
    redis_settings = REDIS_SETTINGS


async def main():
    redis = await create_pool(REDIS_SETTINGS)
    await enqueue_jobs(redis)


if __name__ == '__main__':
    asyncio.run(main())
```

----------------------------------------

TITLE: Install flake8-pydantic Plugin
DESCRIPTION: Installs the flake8-pydantic plugin using pip. This plugin provides linting capabilities for Pydantic models within your project.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/integrations/linting.md#_snippet_0

LANGUAGE: bash
CODE:
```
pip install flake8-pydantic
```

----------------------------------------

TITLE: Mypy Configuration File Example
DESCRIPTION: An example of a Mypy configuration file (`.ini` format) used by the test suite. These files specify Mypy settings and plugins to be applied during type checking for specific test cases.

SOURCE: https://github.com/pydantic/pydantic/blob/main/tests/mypy/README.md#_snippet_3

LANGUAGE: ini
CODE:
```
[mypy]
plugins = pydantic.mypy

[mypy "test_mypy.py"]
ignore_missing_imports = true
```

----------------------------------------

TITLE: Pydantic Model Example
DESCRIPTION: Demonstrates a Pydantic BaseModel with various field types and potential validation issues. This snippet is used to illustrate the benefits of the Pydantic mypy plugin.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/integrations/mypy.md#_snippet_0

LANGUAGE: python
CODE:
```
from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class Model(BaseModel):
    age: int
    first_name = 'John'
    last_name: Optional[str] = None
    signup_ts: Optional[datetime] = None
    list_of_ints: list[int]


m = Model(age=42, list_of_ints=[1, '2', b'3'])
print(m.middle_name)  # not a model field!
Model()  # will raise a validation error for age and list_of_ints
```

----------------------------------------

TITLE: Python Docstring Example: Function
DESCRIPTION: Illustrates the Google-style docstring format for a Python function, detailing arguments and return values. This adheres to PEP 257 and is checked by pydocstyle for consistency.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/contributing.md#_snippet_10

LANGUAGE: python
CODE:
```
def bar(self, baz: int) -> str:
    """A function docstring.

    Args:
        baz: A description of `baz`.

    Returns:
        A description of the return value.
    """

    return 'bar'
```

----------------------------------------

TITLE: Pydantic BaseModel Example
DESCRIPTION: Demonstrates creating a Pydantic model with type hints for validation and serialization. Shows data coercion and model dumping.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/index.md#_snippet_1

LANGUAGE: python
CODE:
```
from datetime import datetime

from pydantic import BaseModel, PositiveInt


class User(BaseModel):
    id: int
    name: str = 'John Doe'
    signup_ts: datetime | None
    tastes: dict[str, PositiveInt]


external_data = {
    'id': 123,
    'signup_ts': '2019-06-01 12:22',
    'tastes': {
        'wine': 9,
        b'cheese': 7,
        'cabbage': '1',
    },
}

user = User(**external_data)

print(user.id)
print(user.model_dump())
```

----------------------------------------

TITLE: Pydantic Data Model Example (Python)
DESCRIPTION: Demonstrates creating a Pydantic BaseModel for user data. It shows how to define fields with type hints, default values, and optional types. The example validates external data, converting types and handling missing values.

SOURCE: https://github.com/pydantic/pydantic/blob/main/README.md#_snippet_0

LANGUAGE: Python
CODE:
```
from datetime import datetime
from typing import Optional
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str = 'John Doe'
    signup_ts: Optional[datetime] = None
    friends: list[int] = []

external_data = {'id': '123', 'signup_ts': '2017-06-01 12:22', 'friends': [1, '2', b'3']}
user = User(**external_data)
print(user)
#> User id=123 name='John Doe' signup_ts=datetime.datetime(2017, 6, 1, 12, 22) friends=[1, 2, 3]
print(user.id)
#> 123
```

----------------------------------------

TITLE: Install Pydantic for AWS Lambda
DESCRIPTION: Installs the Pydantic library for AWS Lambda functions using pip. This command specifies platform compatibility (manylinux2014_x86_64), a target directory for packaging, the CPython implementation, a compatible Python version (3.10), and ensures pre-built binary wheels are used. This is crucial for ensuring compatibility between your local development environment and the AWS Lambda runtime.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/integrations/aws_lambda.md#_snippet_0

LANGUAGE: bash
CODE:
```
pip install \
    --platform manylinux2014_x86_64 \  # (1)!
    --target=<your_package_dir> \  # (2)!
    --implementation cp \  # (3)!
    --python-version 3.10 \  # (4)!
    --only-binary=:all: \  # (5)!
    --upgrade pydantic  # (6)!
```

----------------------------------------

TITLE: Get Pydantic Version String
DESCRIPTION: Access the primary version string of the Pydantic library. This is typically a simple string representation of the installed version.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/version.md#_snippet_0

LANGUAGE: python
CODE:
```
import pydantic

print(pydantic.__version__)
```

----------------------------------------

TITLE: Python Docstring Example: Class
DESCRIPTION: Demonstrates the correct Google-style docstring format for a Python class, including documentation for class attributes. This follows PEP 257 guidelines and is linted by pydocstyle.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/contributing.md#_snippet_9

LANGUAGE: python
CODE:
```
class Foo:
    """A class docstring.

    Attributes:
        bar: A description of bar. Defaults to "bar".
    """

    bar: str = 'bar'
```

----------------------------------------

TITLE: Get Pydantic Version (pre-v2)
DESCRIPTION: This command is used to get Pydantic version information for versions prior to v2.0. It's essential for users on older Pydantic versions when reporting issues.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/contributing.md#_snippet_1

LANGUAGE: python
CODE:
```
import pydantic.utils; print(pydantic.utils.version_info())
```

----------------------------------------

TITLE: Pydantic `__pydantic_on_complete__()` Hook
DESCRIPTION: Introduces a new hook, `__pydantic_on_complete__()`, which is executed once a Pydantic model is fully ready and all its fields are complete. This hook is useful for performing final setup or validation steps after a model has been initialized.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_0

LANGUAGE: python
CODE:
```
class MyModel:
    # ... model definition ...

    def __pydantic_on_complete__(self):
        # This method is called after the model is fully ready
        print("Model is complete and ready to use!")

```

----------------------------------------

TITLE: BaseModel Instantiated Directly: Python Example
DESCRIPTION: This error occurs when `BaseModel` is instantiated directly without inheriting from it. The example shows how to catch this specific PydanticUserError.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/usage_errors.md#_snippet_22

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, PydanticUserError

try:
    BaseModel()
except PydanticUserError as exc_info:
    assert exc_info.code == 'base-model-instantiated'
```

----------------------------------------

TITLE: Pydantic Documentation Syntax Highlighting
DESCRIPTION: Fixes local syntax highlighting issues within the documentation extensions. This ensures that code examples in the documentation are displayed correctly.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_161

LANGUAGE: APIDOC
CODE:
```
Pydantic Documentation Syntax Highlighting:

Updated docs extensions to fix local syntax highlighting.
```

----------------------------------------

TITLE: Config and model_config Both Defined: Python Example
DESCRIPTION: This error occurs when both the legacy `class Config` and the modern `model_config` are defined within the same Pydantic model. The example illustrates catching this conflict.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/usage_errors.md#_snippet_18

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ConfigDict, PydanticUserError

try:

    class Model(BaseModel):
        model_config = ConfigDict(from_attributes=True)

        a: str

        class Config:
            from_attributes = True

except PydanticUserError as exc_info:
    assert exc_info.code == 'config-both'
```

----------------------------------------

TITLE: Basic Pydantic Logging with Logfire
DESCRIPTION: Demonstrates how to configure Logfire and log a Pydantic BaseModel instance. This snippet shows the basic setup for sending Pydantic model data to Logfire for observability.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/integrations/logfire.md#_snippet_0

LANGUAGE: python
CODE:
```
from datetime import date

import logfire

from pydantic import BaseModel

logfire.configure()  # (1)!


class User(BaseModel):
    name: str
    country_code: str
    dob: date


user = User(name='Anne', country_code='USA', dob='2000-01-01')
logfire.info('user processed: {user!r}', user=user)  # (2)!
```

----------------------------------------

TITLE: Use Bump Pydantic Tool
DESCRIPTION: Demonstrates the command-line usage of the 'bump-pydantic' tool. Navigate to your project's root directory and specify the package name to initiate the migration process.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_2

LANGUAGE: bash
CODE:
```
cd /path/to/repo_folder
bump-pydantic my_package
```

----------------------------------------

TITLE: Pydantic JSON Schema Generation Example
DESCRIPTION: Illustrates how to generate a JSON Schema from a Pydantic model, which is useful for self-documenting APIs and integrating with tools that support the JSON Schema format.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/why.md#_snippet_3

LANGUAGE: python
CODE:
```
from datetime import datetime

from pydantic import BaseModel


class Address(BaseModel):
    street: str
    city: str
    zipcode: str


class Meeting(BaseModel):
    when: datetime
    where: Address
    why: str = 'No idea'


print(Meeting.model_json_schema())
"""
{
    '$defs': {
        'Address': {
            'properties': {
                'street': {'title': 'Street', 'type': 'string'},
                'city': {'title': 'City', 'type': 'string'},
                'zipcode': {'title': 'Zipcode', 'type': 'string'},
            },
            'required': ['street', 'city', 'zipcode'],
            'title': 'Address',
            'type': 'object',
        }
    },
    'properties': {
        'when': {'format': 'date-time', 'title': 'When', 'type': 'string'},
        'where': {'$ref': '#/$defs/Address'},
        'why': {'default': 'No idea', 'title': 'Why', 'type': 'string'},
    },
    'required': ['when', 'where'],
    'title': 'Meeting',
    'type': 'object',
}
"""
```

----------------------------------------

TITLE: Undefined Annotation: Python Example
DESCRIPTION: This error is raised when Pydantic encounters an undefined annotation during schema generation. The example shows how to catch `PydanticUndefinedAnnotation` for a forward-referenced type.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/usage_errors.md#_snippet_23

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, PydanticUndefinedAnnotation


class Model(BaseModel):
    a: 'B'  # noqa F821


try:
    Model.model_rebuild()
except PydanticUndefinedAnnotation as exc_info:
    assert exc_info.code == 'undefined-annotation'
```

----------------------------------------

TITLE: Pydantic Time Parsing Example
DESCRIPTION: Demonstrates Pydantic's capability to parse time strings into Python's datetime.time objects.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/standard_library_types.md#_snippet_3

LANGUAGE: python
CODE:
```
from datetime import time

from pydantic import BaseModel


class Meeting(BaseModel):
    t: time = None


m = Meeting(t=time(4, 8, 16))

print(m.model_dump())
#> {'t': datetime.time(4, 8, 16)}

```

----------------------------------------

TITLE: Pydantic Data Conversion Example
DESCRIPTION: Demonstrates how Pydantic automatically casts input data to conform to model field types, potentially leading to information loss. Includes an example of using strict mode for type enforcement.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_9

LANGUAGE: python
CODE:
```
from pydantic import BaseModel


class Model(BaseModel):
    a: int
    b: float
    c: str


print(Model(a=3.000, b='2.72', c=b'binary data').model_dump())
#> {'a': 3, 'b': 2.72, 'c': 'binary data'}


class ModelWithList(BaseModel):
    items: list[int]


print(ModelWithList(items=(1, 2, 3)))
#> items=[1, 2, 3]
```

----------------------------------------

TITLE: Pydantic BaseModel Example for Mypy Testing
DESCRIPTION: Illustrates a Pydantic `BaseModel` definition and its instantiation with an extra keyword argument. This serves as a typical input file for the Mypy test suite, demonstrating a scenario that Mypy would analyze for type checking errors.

SOURCE: https://github.com/pydantic/pydantic/blob/main/tests/mypy/README.md#_snippet_0

LANGUAGE: python
CODE:
```
from pydantic import BaseModel


class Model(BaseModel):
    a: int


model = Model(a=1, b=2)
```

----------------------------------------

TITLE: Pydantic Strict Mode and Data Coercion Example
DESCRIPTION: Shows how Pydantic handles data validation, contrasting default type coercion with strict mode, and demonstrates parsing JSON data with type conversion.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/why.md#_snippet_4

LANGUAGE: python
CODE:
```
from datetime import datetime

from pydantic import BaseModel, ValidationError


class Meeting(BaseModel):
    when: datetime
    where: bytes


m = Meeting.model_validate({'when': '2020-01-01T12:00', 'where': 'home'})
print(m)
#> when=datetime.datetime(2020, 1, 1, 12, 0) where=b'home'
try:
    m = Meeting.model_validate(
        {'when': '2020-01-01T12:00', 'where': 'home'},
        strict=True
    )
except ValidationError as e:
    print(e)
    """
2 validation errors for Meeting
when
  Input should be a valid datetime [type=datetime_type, input_value='2020-01-01T12:00', input_type=str]
where
  Input should be a valid bytes [type=bytes_type, input_value='home', input_type=str]
"""

m_json = Meeting.model_validate_json(
    '{"when": "2020-01-01T12:00", "where": "home"}'
)
print(m_json)
#> when=datetime.datetime(2020, 1, 1, 12, 0) where=b'home'
```

----------------------------------------

TITLE: Clone Pydantic Repository
DESCRIPTION: Instructions to clone your fork of the Pydantic repository from GitHub and navigate into the project directory. This is the first step in setting up a local development environment.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/contributing.md#_snippet_2

LANGUAGE: bash
CODE:
```
git clone git@github.com:<your username>/pydantic.git
cd pydantic
```

----------------------------------------

TITLE: Pydantic Model Validation Examples
DESCRIPTION: Demonstrates the usage of Pydantic's model_validate, model_validate_json, and model_validate_strings methods with various inputs, including successful validations and error handling for invalid data.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_21

LANGUAGE: python
CODE:
```
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ValidationError


class User(BaseModel):
    id: int
    name: str = 'John Doe'
    signup_ts: Optional[datetime] = None


# Example 1: model_validate with a dictionary
m = User.model_validate({'id': 123, 'name': 'James'})
print(m)
#> id=123 name='James' signup_ts=None

# Example 2: model_validate with invalid input type
try:
    User.model_validate(['not', 'a', 'dict'])
except ValidationError as e:
    print(e)
    """
    1 validation error for User
      Input should be a valid dictionary or instance of User [type=model_type, input_value=['not', 'a', 'dict'], input_type=list]
    """

# Example 3: model_validate_json with valid JSON string
m = User.model_validate_json('{"id": 123, "name": "James"}')
print(m)
#> id=123 name='James' signup_ts=None

# Example 4: model_validate_json with invalid data type in JSON
try:
    m = User.model_validate_json('{"id": 123, "name": 123}')
except ValidationError as e:
    print(e)
    """
    1 validation error for User
    name
      Input should be a valid string [type=string_type, input_value=123, input_type=int]
    """

# Example 5: model_validate_json with invalid JSON format
try:
    m = User.model_validate_json('invalid JSON')
except ValidationError as e:
    print(e)
    """
    1 validation error for User
      Invalid JSON: expected value at line 1 column 1 [type=json_invalid, input_value='invalid JSON', input_type=str]
    """

# Example 6: model_validate_strings with string values
m = User.model_validate_strings({'id': '123', 'name': 'James'})
print(m)
#> id=123 name='James' signup_ts=None

# Example 7: model_validate_strings with datetime string
m = User.model_validate_strings(
    {'id': '123', 'name': 'James', 'signup_ts': '2024-04-01T12:00:00'}
)
print(m)
#> id=123 name='James' signup_ts=datetime.datetime(2024, 4, 1, 12, 0)

# Example 8: model_validate_strings with strict=True and invalid datetime format
try:
    m = User.model_validate_strings(
        {'id': '123', 'name': 'James', 'signup_ts': '2024-04-01'},
        strict=True
    )
except ValidationError as e:
    print(e)
    """
    1 validation error for User
    signup_ts
      Input should be a valid datetime, invalid datetime separator, expected `T`, `t`, `_` or space [type=datetime_parsing, input_value='2024-04-01', input_type=str]
    """

```

----------------------------------------

TITLE: Format Full Changelog Link for GitHub Release
DESCRIPTION: Defines the format for the full changelog link to be included in the GitHub release body, comparing the previous and current versions.

SOURCE: https://github.com/pydantic/pydantic/blob/main/release/README.md#_snippet_5

LANGUAGE: Markdown
CODE:
```
Full Changelog: https://github.com/pydantic/pydantic/compare/v{PREV_VERSION}...v{VERSION}/
```

----------------------------------------

TITLE: Create GitHub Release Tag and Body
DESCRIPTION: Creates a new release on GitHub. This involves setting the tag to `v{VERSION}`, the title to `v{VERSION} {DATE}`, and populating the body with the prepared `HISTORY.md` section and a full changelog link.

SOURCE: https://github.com/pydantic/pydantic/blob/main/release/README.md#_snippet_4

LANGUAGE: shell
CODE:
```
git tag v{VERSION}
git push origin v{VERSION}
```

----------------------------------------

TITLE: Mapping validate_as to Validator Types
DESCRIPTION: Shows how the `validate_as` method in the pipeline API maps to Pydantic's `BeforeValidator`, `AfterValidator`, and `WrapValidator`. It provides examples for pre-processing, post-processing, and wrapping validation logic.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/experimental.md#_snippet_2

LANGUAGE: python
CODE:
```
from typing import Annotated

from pydantic.experimental.pipeline import transform, validate_as

# BeforeValidator
Annotated[int, validate_as(str).str_strip().validate_as(...)]  # (1)!
# AfterValidator
Annotated[int, transform(lambda x: x * 2)]  # (2)!
# WrapValidator
Annotated[
    int,
    validate_as(str)
    .str_strip()
    .validate_as(...)
    .transform(lambda x: x * 2),  # (3)!
]
```

----------------------------------------

TITLE: Pydantic Model Serialization Examples
DESCRIPTION: Demonstrates serializing a Pydantic model to a Python dict (with Python objects), a JSONable dict, and a JSON string, showcasing options like excluding unset or default fields.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/why.md#_snippet_2

LANGUAGE: python
CODE:
```
from datetime import datetime

from pydantic import BaseModel


class Meeting(BaseModel):
    when: datetime
    where: bytes
    why: str = 'No idea'


m = Meeting(when='2020-01-01T12:00', where='home')
print(m.model_dump(exclude_unset=True))
#> {'when': datetime.datetime(2020, 1, 1, 12, 0), 'where': b'home'}
print(m.model_dump(exclude={'where'}, mode='json'))
#> {'when': '2020-01-01T12:00:00', 'why': 'No idea'}
print(m.model_dump_json(exclude_defaults=True))
#> {"when":"2020-01-01T12:00:00","where":"home"}
```

----------------------------------------

TITLE: Keyword Arguments Removed (regex): Python Example
DESCRIPTION: This error indicates that certain keyword arguments, like `regex`, have been removed in Pydantic V2. The example shows a V1-style usage that would trigger this error.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/usage_errors.md#_snippet_19

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field, PydanticUserError

try:

    class Model(BaseModel):
        x: str = Field(regex='test')

except PydanticUserError as exc_info:
    assert exc_info.code == 'removed-kwargs'
```

----------------------------------------

TITLE: Instantiate and Print Model with Generic Owners
DESCRIPTION: Provides an example of creating an instance of the `Model` class, populating it with `Owner` objects containing specific types (`Car`, `House`), and printing the resulting model. This demonstrates successful instantiation.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/types.md#_snippet_20

LANGUAGE: python
CODE:
```
model = Model(
    car_owner=Owner(name='John', item=Car(color='black')),
    home_owner=Owner(name='James', item=House(rooms=3)),
)
print(model)

```

----------------------------------------

TITLE: datetime_past Pydantic Validation Example
DESCRIPTION: Shows the 'datetime_past' error, triggered when a value assigned to a PastDatetime field is not in the past. The example creates a datetime object in the future.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_23

LANGUAGE: python
CODE:
```
from datetime import datetime, timedelta

from pydantic import BaseModel, PastDatetime, ValidationError


class Model(BaseModel):
    x: PastDatetime


try:
    Model(x=datetime.now() + timedelta(100))
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'datetime_past'
```

----------------------------------------

TITLE: decimal_parsing Pydantic Validation Example
DESCRIPTION: Demonstrates the 'decimal_parsing' error, which occurs when a value cannot be parsed into a Decimal number. The example attempts to parse the string 'test' into a Decimal field.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_27

LANGUAGE: python
CODE:
```
from decimal import Decimal

from pydantic import BaseModel, Field, ValidationError


class Model(BaseModel):
    x: Decimal = Field(decimal_places=3)


try:
    Model(x='test')
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'decimal_parsing'
```

----------------------------------------

TITLE: Prepare Release with Python Script
DESCRIPTION: Runs the release preparation script from the repository root. This script updates the version number in `version.py`, runs `uv lock`, and adds a new section to `HISTORY.md`. A `--dry-run` flag can be used to preview changes without modifying files.

SOURCE: https://github.com/pydantic/pydantic/blob/main/release/README.md#_snippet_1

LANGUAGE: shell
CODE:
```
uv run release/prepare.py {VERSION}
```

----------------------------------------

TITLE: Validate INI Data with Pydantic
DESCRIPTION: Shows how to load and validate data from an INI configuration file using Python's `configparser` module and a Pydantic `BaseModel`. The example defines a `Person` model and validates data from a specific section of the INI file.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/examples/files.md#_snippet_12

LANGUAGE: ini
CODE:
```
[PERSON]
name = John Doe
age = 30
email = john@example.com
```

LANGUAGE: python
CODE:
```
import configparser

from pydantic import BaseModel, EmailStr, PositiveInt


class Person(BaseModel):
    name: str
    age: PositiveInt
    email: EmailStr


config = configparser.ConfigParser()
config.read('person.ini')
person = Person.model_validate(config['PERSON'])
print(person)
#> name='John Doe' age=30 email='john@example.com'
```

----------------------------------------

TITLE: Model Field Overridden: Python Example
DESCRIPTION: This error is raised when a field defined on a base class was overridden by a non-annotated attribute. The example demonstrates catching this specific PydanticUserError.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/usage_errors.md#_snippet_14

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, PydanticUserError


class Foo(BaseModel):
    a: float


try:

    class Bar(Foo):
        x: float = 12.3
        a = 123.0

except PydanticUserError as exc_info:
    assert exc_info.code == 'model-field-overridden'
```

----------------------------------------

TITLE: Mypy Configuration for Pydantic Plugin
DESCRIPTION: Shows how to enable the Pydantic mypy plugin by adding 'pydantic.mypy' to the plugins list in mypy configuration files.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/integrations/mypy.md#_snippet_1

LANGUAGE: ini
CODE:
```
[mypy]
plugins = pydantic.mypy
```

LANGUAGE: toml
CODE:
```
[tool.mypy]
plugins = ['pydantic.mypy']
```

----------------------------------------

TITLE: Pydantic init_typed Example
DESCRIPTION: Demonstrates how Pydantic's default data conversion allows string input for integer fields. The `init_typed` plugin setting prevents this by synthesizing `__init__` with explicit type annotations for fields.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/integrations/mypy.md#_snippet_4

LANGUAGE: python
CODE:
```
class Model(BaseModel):
    a: int


Model(a='1')
```

----------------------------------------

TITLE: Pydantic Date Parsing Example
DESCRIPTION: Shows how Pydantic can convert Unix timestamps (integers or floats) and date strings into Python's datetime.date objects.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/standard_library_types.md#_snippet_2

LANGUAGE: python
CODE:
```
from datetime import date

from pydantic import BaseModel


class Birthday(BaseModel):
    d: date = None


my_birthday = Birthday(d=1679616000.0)

print(my_birthday.model_dump())
#> {'d': datetime.date(2023, 3, 24)}

```

----------------------------------------

TITLE: Build Documentation
DESCRIPTION: Builds the project's documentation using mkdocs-material. This command is used to verify that any documentation changes you've made render correctly.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/contributing.md#_snippet_8

LANGUAGE: bash
CODE:
```
make docs
```

----------------------------------------

TITLE: Pydantic: ContextVar for Model Instantiation with Context
DESCRIPTION: Illustrates a workaround for passing context during direct Pydantic model instantiation using `ContextVar` and a custom `__init__`. This enables context-aware validation when creating model instances, requiring `pydantic`, `contextvars`, and `typing`. The example shows multiplying a number by a context-provided multiplier.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/validators.md#_snippet_18

LANGUAGE: python
CODE:
```
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Generator

from pydantic import BaseModel, ValidationInfo, field_validator

_init_context_var = ContextVar('_init_context_var', default=None)


@contextmanager
def init_context(value: dict[str, Any]) -> Generator[None]:
    token = _init_context_var.set(value)
    try:
        yield
    finally:
        _init_context_var.reset(token)


class Model(BaseModel):
    my_number: int

    def __init__(self, /, **data: Any) -> None:
        self.__pydantic_validator__.validate_python(
            data,
            self_instance=self,
            context=_init_context_var.get(),
        )

    @field_validator('my_number')
    @classmethod
    def multiply_with_context(cls, value: int, info: ValidationInfo) -> int:
        if isinstance(info.context, dict):
            multiplier = info.context.get('multiplier', 1)
            value = value * multiplier
        return value


print(Model(my_number=2))
#> my_number=2

with init_context({'multiplier': 3}):
    print(Model(my_number=2))
    #> my_number=6

print(Model(my_number=2))
#> my_number=2
```

----------------------------------------

TITLE: Schema for Unknown Type: Python Example
DESCRIPTION: This error occurs when Pydantic fails to generate a schema for an unknown or unsupported type. The example shows a model with an integer literal as a type annotation, triggering the error.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/usage_errors.md#_snippet_24

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, PydanticUserError

try:

    class Model(BaseModel):
        x: 43 = 123

except PydanticUserError as exc_info:
    assert exc_info.code == 'schema-for-unknown-type'
```

----------------------------------------

TITLE: Get Detailed Pydantic Version Info
DESCRIPTION: Retrieve more detailed version information for Pydantic, which might include build numbers, commit hashes, or other version-related metadata.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/version.md#_snippet_1

LANGUAGE: python
CODE:
```
from pydantic.version import version_info

print(version_info())
```

----------------------------------------

TITLE: enum Pydantic Validation Example
DESCRIPTION: Illustrates the 'enum' error, which occurs when an input value does not match any of the members in an Enum field. The example uses a string Enum and provides an invalid option.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_31

LANGUAGE: python
CODE:
```
from enum import Enum

from pydantic import BaseModel, ValidationError


class MyEnum(str, Enum):
    option = 'option'


class Model(BaseModel):
    x: MyEnum


try:
    Model(x='other_option')
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'enum'
```

----------------------------------------

TITLE: Get Pydantic Version (v2+)
DESCRIPTION: This command retrieves the Pydantic version information, which is crucial for reporting bugs or issues. It executes a Python script to print the version details.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/contributing.md#_snippet_0

LANGUAGE: python
CODE:
```
import pydantic.version; print(pydantic.version.version_info())
```

----------------------------------------

TITLE: JSON Schema Invalid Type: Python Example
DESCRIPTION: This error is raised when Pydantic encounters a type it cannot convert into a JSON schema, such as `ImportString` in this example. The code demonstrates triggering and catching this error.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/usage_errors.md#_snippet_21

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ImportString, PydanticUserError


class Model(BaseModel):
    a: ImportString


try:
    Model.model_json_schema()
except PydanticUserError as exc_info:
    assert exc_info.code == 'invalid-for-json-schema'
```

----------------------------------------

TITLE: dict_type Pydantic Validation Example
DESCRIPTION: Demonstrates the 'dict_type' error, raised when the input value's type is not a dictionary for a dict field. The example attempts to assign a list to a dictionary field.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_30

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ValidationError


class Model(BaseModel):
    x: dict


try:
    Model(x=['1', '2'])
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'dict_type'
```

----------------------------------------

TITLE: Check `pydantic_core` Files (Python)
DESCRIPTION: This Python snippet uses `importlib.metadata` to list files within the `pydantic-core` package. It helps verify if the compiled library and type stubs, specifically `_pydantic_core`, are present, which is crucial for correct Pydantic installation.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/integrations/aws_lambda.md#_snippet_1

LANGUAGE: python
CODE:
```
from importlib.metadata import files
print([file for file in files('pydantic-core') if file.name.startswith('_pydantic_core')])
"""
[PackagePath('pydantic_core/_pydantic_core.pyi'), PackagePath('pydantic_core/_pydantic_core.cpython-312-x86_64-linux-gnu.so')]
"""
```

----------------------------------------

TITLE: Pydantic Model Pickling Support
DESCRIPTION: Illustrates Pydantic models' support for efficient pickling and unpickling. This allows models to be serialized and deserialized, preserving their state. The example shows the process of pickling a model instance and then unpickling it.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/serialization.md#_snippet_3

LANGUAGE: python
CODE:
```
import pickle

from pydantic import BaseModel


class FooBarModel(BaseModel):
    a: str
    b: int


m = FooBarModel(a='hello', b=123)
print(m)
#> a='hello' b=123
data = pickle.dumps(m)
print(data[:20])
#> b'\x80\x04\x95\x95\x00\x00\x00\x00\x00\x00\x00\x8c\x08__main_'
m2 = pickle.loads(data)
print(m2)
#> a='hello' b=123
```

----------------------------------------

TITLE: datetime_parsing Pydantic Validation Example
DESCRIPTION: Illustrates the 'datetime_parsing' error, which occurs when a string value cannot be parsed into a datetime field. The example uses a strict datetime field and invalid JSON input.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_22

LANGUAGE: python
CODE:
```
import json
from datetime import datetime

from pydantic import BaseModel, Field, ValidationError


class Model(BaseModel):
    x: datetime = Field(strict=True)


try:
    Model.model_validate_json(json.dumps({'x': 'not a datetime'}))
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'datetime_parsing'
```

----------------------------------------

TITLE: Pydantic Configuration API
DESCRIPTION: Documentation for Pydantic's configuration system, including ConfigDict, with_config, ExtraValues, and BaseConfig. This section details how to manage model configuration and settings.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/config.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
pydantic.config:
  members:
    - ConfigDict
    - with_config
    - ExtraValues
    - BaseConfig
  options:
    group_by_category: false
```

----------------------------------------

TITLE: `missing_keyword_only_argument` Validation Error Example (Python)
DESCRIPTION: This error is raised when a required keyword-only argument is not provided to a function decorated with `validate_call`. The example defines a function `foo` with a keyword-only argument `a` and calls it without passing `a`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_60

LANGUAGE: python
CODE:
```
from pydantic import ValidationError, validate_call


@validate_call
def foo(*, a: int):
    return a


try:
    foo()
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'missing_keyword_only_argument'
```

----------------------------------------

TITLE: decimal_whole_digits Pydantic Validation Example
DESCRIPTION: Shows the 'decimal_whole_digits' error, triggered when a Decimal value has more digits before the decimal point than allowed by the combined max_digits and decimal_places constraints. The example uses max_digits=6 and decimal_places=3.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_29

LANGUAGE: python
CODE:
```
from decimal import Decimal

from pydantic import BaseModel, Field, ValidationError


class Model(BaseModel):
    x: Decimal = Field(max_digits=6, decimal_places=3)


try:
    Model(x='12345.6')
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'decimal_whole_digits'
```

----------------------------------------

TITLE: Pydantic Boolean Validation Example
DESCRIPTION: Demonstrates Pydantic's flexible boolean validation, accepting various string representations, integers 0/1, and standard booleans. It also shows how a ValidationError is raised for invalid inputs.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/standard_library_types.md#_snippet_0

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ValidationError


class BooleanModel(BaseModel):
    bool_value: bool


print(BooleanModel(bool_value=False))
#> bool_value=False
print(BooleanModel(bool_value='False'))
#> bool_value=False
print(BooleanModel(bool_value=1))
#> bool_value=True
try:
    BooleanModel(bool_value=[])
except ValidationError as e:
    print(str(e))
    """
    1 validation error for BooleanModel
    bool_value
      Input should be a valid boolean [type=bool_type, input_value=[], input_type=list]
    """

```

----------------------------------------

TITLE: decimal_max_digits Pydantic Validation Example
DESCRIPTION: Illustrates the 'decimal_max_digits' error, which occurs when a Decimal value exceeds the specified maximum number of digits. The example sets max_digits to 3 and provides a value with more digits.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_25

LANGUAGE: python
CODE:
```
from decimal import Decimal

from pydantic import BaseModel, Field, ValidationError


class Model(BaseModel):
    x: Decimal = Field(max_digits=3)


try:
    Model(x='42.1234')
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'decimal_max_digits'
```

----------------------------------------

TITLE: Validate YAML Data with Pydantic
DESCRIPTION: Illustrates how to load and validate data from a YAML file using the `PyYAML` library and a Pydantic `BaseModel`. The example defines a `Person` model and validates the loaded YAML data.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/examples/files.md#_snippet_10

LANGUAGE: yaml
CODE:
```
name: John Doe
age: 30
email: john@example.com
```

LANGUAGE: python
CODE:
```
import yaml

from pydantic import BaseModel, EmailStr, PositiveInt


class Person(BaseModel):
    name: str
    age: PositiveInt
    email: EmailStr


with open('person.yaml') as f:
    data = yaml.safe_load(f)

person = Person.model_validate(data)
print(person)
#> name='John Doe' age=30 email='john@example.com'
```

----------------------------------------

TITLE: Generate JSON Schema for Boolean
DESCRIPTION: Demonstrates the JSON Schema generated for a boolean type by Pydantic's GenerateJsonSchema class, starting from a core schema.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/internals/architecture.md#_snippet_5

LANGUAGE: json
CODE:
```
{
    "type": "boolean"
}
```

----------------------------------------

TITLE: `missing_positional_only_argument` Validation Error Example (Python)
DESCRIPTION: This error occurs when a required positional-only argument is not passed to a function decorated with `validate_call`. The example defines a function `foo` with a positional-only argument `a` and calls it without providing `a`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_61

LANGUAGE: python
CODE:
```
from pydantic import ValidationError, validate_call


@validate_call
def foo(a: int, /):
    return a


try:
    foo()
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'missing_positional_only_argument'
```

----------------------------------------

TITLE: Pydantic Timedelta Parsing Example
DESCRIPTION: Illustrates Pydantic's support for parsing timedelta values from integers, floats (seconds), and various string formats including ISO 8601.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/standard_library_types.md#_snippet_4

LANGUAGE: python
CODE:
```
from datetime import timedelta

from pydantic import BaseModel


class Model(BaseModel):
    td: timedelta = None


m = Model(td='P3DT12H30M5S')

print(m.model_dump())
#> {'td': datetime.timedelta(days=3, seconds=45005)}

```

----------------------------------------

TITLE: String Constraints with Pydantic
DESCRIPTION: Demonstrates how to use Pydantic's Field to enforce minimum length, maximum length, and regular expression patterns on string fields within a BaseModel. Includes an example of model instantiation and output.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/fields.md#_snippet_20

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field


class Foo(BaseModel):
    short: str = Field(min_length=3)
    long: str = Field(max_length=10)
    regex: str = Field(pattern=r'^\d*$')  # (1)!


foo = Foo(short='foo', long='foobarbaz', regex='123')
print(foo)
#> short='foo' long='foobarbaz' regex='123'
```

----------------------------------------

TITLE: Importing Experimental Features in Pydantic
DESCRIPTION: Demonstrates how to import experimental features from the pydantic.experimental module and how to suppress the associated experimental warning.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/version-policy.md#_snippet_0

LANGUAGE: python
CODE:
```
from pydantic.experimental import feature_name
```

LANGUAGE: python
CODE:
```
import warnings

from pydantic import PydanticExperimentalWarning

warnings.filterwarnings('ignore', category=PydanticExperimentalWarning)
```

----------------------------------------

TITLE: Pydantic User Data Readability
DESCRIPTION: This Python snippet illustrates a Pydantic User object with nested Address details. It emphasizes Pydantic's ability to create more readable and maintainable data models compared to plain string representations.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/plugins/devtools_output.html#_snippet_0

LANGUAGE: python
CODE:
```
user: User(
    id=123,
    name='John Doe',
    signup_ts=datetime.datetime(2019, 6, 1, 12, 22),
    friends=[ 1234, 4567, 7890, ],
    address=Address(
        street='Testing',
        country='uk',
        lat=51.5,
        lng=0.0,
    ),
)
```

----------------------------------------

TITLE: `missing_argument` Validation Error Example (Python)
DESCRIPTION: This error occurs when a required positional-or-keyword argument is not passed to a function decorated with `validate_call`. The example defines a function `foo` requiring an argument `a` and calls it without any arguments.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_59

LANGUAGE: python
CODE:
```
from pydantic import ValidationError, validate_call


@validate_call
def foo(a: int):
    return a


try:
    foo()
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'missing_argument'
```

----------------------------------------

TITLE: decimal_max_places Pydantic Validation Example
DESCRIPTION: Shows the 'decimal_max_places' error, raised when a Decimal value has more digits after the decimal point than allowed. The example sets decimal_places to 3 and provides a value with four decimal places.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_26

LANGUAGE: python
CODE:
```
from decimal import Decimal

from pydantic import BaseModel, Field, ValidationError


class Model(BaseModel):
    x: Decimal = Field(decimal_places=3)


try:
    Model(x='42.1234')
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'decimal_max_places'
```

----------------------------------------

TITLE: SerializeAsAny Annotation Example
DESCRIPTION: Demonstrates using the SerializeAsAny annotation to achieve duck typing serialization behavior for a Pydantic model field. This allows a field to be serialized as if its type hint was Any.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/serialization.md#_snippet_17

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, SerializeAsAny


class User(BaseModel):
    name: str


class UserLogin(User):
    password: str


class OuterModel(BaseModel):
    as_any: SerializeAsAny[User]
    as_user: User


user = UserLogin(name='pydantic', password='password')

print(OuterModel(as_any=user, as_user=user).model_dump())
```

----------------------------------------

TITLE: Pydantic Model Copying Example
DESCRIPTION: Demonstrates how to duplicate Pydantic models using the `model_copy()` method, including options for updating fields and performing deep copies. Shows how `deep=True` affects nested model references.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_23

LANGUAGE: python
CODE:
```
from pydantic import BaseModel


class BarModel(BaseModel):
    whatever: int


class FooBarModel(BaseModel):
    banana: float
    foo: str
    bar: BarModel


m = FooBarModel(banana=3.14, foo='hello', bar={'whatever': 123})

print(m.model_copy(update={'banana': 0}))
#> banana=0 foo='hello' bar=BarModel(whatever=123)

# normal copy gives the same object reference for bar:
print(id(m.bar) == id(m.model_copy().bar))
#> True
# deep copy gives a new object reference for `bar`:
print(id(m.bar) == id(m.model_copy(deep=True).bar))
#> False
```

----------------------------------------

TITLE: Pydantic UserIn/UserOut Data Transformation
DESCRIPTION: This Python snippet defines Pydantic models `UserIn` and `UserOut` to handle data validation and transformation. It includes a function `my_api` that processes user input, converting a string representation of a number to an integer, and returns a validated output model. The example demonstrates idiomatic Python for data handling.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/experimental.md#_snippet_3

LANGUAGE: python
CODE:
```
from __future__ import annotations

from pydantic import BaseModel


class UserIn(BaseModel):
    favorite_number: int | str


class UserOut(BaseModel):
    favorite_number: int


def my_api(user: UserIn) -> UserOut:
    favorite_number = user.favorite_number
    if isinstance(favorite_number, str):
        favorite_number = int(user.favorite_number.strip())

    return UserOut(favorite_number=favorite_number)


assert my_api(UserIn(favorite_number=' 1 ')).favorite_number == 1
```

----------------------------------------

TITLE: Pydantic Types API Reference
DESCRIPTION: This section details the Pydantic types module and its documentation rendering options. It specifies how to display root headings and whether to merge initialization methods into class definitions for clarity.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/types.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
pydantic.types
  options:
    show_root_heading: true
    merge_init_into_class: false
```

----------------------------------------

TITLE: AfterValidator Annotated Pattern (Mutate)
DESCRIPTION: Example of using `AfterValidator` with the annotated pattern to mutate the validated value by doubling it. Requires `typing.Annotated` and `pydantic.AfterValidator`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/validators.md#_snippet_2

LANGUAGE: Python
CODE:
```
from typing import Annotated

from pydantic import AfterValidator, BaseModel


def double_number(value: int) -> int:
    return value * 2


class Model(BaseModel):
    number: Annotated[int, AfterValidator(double_number)]


print(Model(number=2))

```

----------------------------------------

TITLE: `missing` Validation Error Example (Python)
DESCRIPTION: This error is raised when required fields are missing from the input value provided for a Pydantic model. The example shows a model with a required field 'x' and attempts to instantiate it without providing any arguments.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_58

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ValidationError


class Model(BaseModel):
    x: str


try:
    Model()
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'missing'
```

----------------------------------------

TITLE: Dataclass Constraints with Pydantic
DESCRIPTION: Demonstrates using Pydantic's Field within dataclasses to control initialization behavior (`init`, `init_var`) and keyword-only arguments (`kw_only`). Includes an example of a nested model structure and its output.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/fields.md#_snippet_23

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass


@dataclass
class Foo:
    bar: str
    baz: str = Field(init_var=True)
    qux: str = Field(kw_only=True)


class Model(BaseModel):
    foo: Foo


model = Model(foo=Foo('bar', baz='baz', qux='qux'))
print(model.model_dump())  # (1)!
#> {'foo': {'bar': 'bar', 'qux': 'qux'}}
```

----------------------------------------

TITLE: Model Field Missing Annotation: Python Example
DESCRIPTION: This error occurs when a field within a Pydantic model lacks a type annotation. The example shows how to catch this error and demonstrates resolving it using ClassVar or ignored_types.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/usage_errors.md#_snippet_15

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field, PydanticUserError

try:

    class Model(BaseModel):
        a = Field('foobar')
        b = None

except PydanticUserError as exc_info:
    assert exc_info.code == 'model-field-missing-annotation'
```

----------------------------------------

TITLE: Pydantic V2: Url Type Behavior Example
DESCRIPTION: Demonstrates the behavior of Pydantic V2's Url types, which use Rust's Url crate and may append slashes to URLs without a path. It shows how to convert Url types to strings.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_31

LANGUAGE: python
CODE:
```
from pydantic import AnyUrl

assert str(AnyUrl(url='https://google.com')) == 'https://google.com/'
assert str(AnyUrl(url='https://google.com/')) == 'https://google.com/'
assert str(AnyUrl(url='https://google.com/api')) == 'https://google.com/api'
assert str(AnyUrl(url='https://google.com/api/')) == 'https://google.com/api/'
```

----------------------------------------

TITLE: Monitor Pydantic with Logfire
DESCRIPTION: Demonstrates how to integrate Pydantic with Logfire to monitor data validations. Logfire records details of both successful and failed validations, providing insights into validation processes. This example shows setting up Logfire, instrumenting Pydantic, and performing validations on a Pydantic model.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/index.md#_snippet_0

LANGUAGE: python
CODE:
```
from datetime import datetime

import logfire

from pydantic import BaseModel

logfire.configure()
logfire.instrument_pydantic()  # (1)!


class Delivery(BaseModel):
    timestamp: datetime
    dimensions: tuple[int, int]


# this will record details of a successful validation to logfire
m = Delivery(timestamp='2020-01-02T03:04:05Z', dimensions=['10', '20'])
print(repr(m.timestamp))
#> datetime.datetime(2020, 1, 2, 3, 4, 5, tzinfo=TzInfo(UTC))
print(m.dimensions)
#> (10, 20)

Delivery(timestamp='2020-01-02T03:04:05Z', dimensions=['10'])  # (2)!
```

----------------------------------------

TITLE: Pydantic: Model Configuration via Class Arguments (Python)
DESCRIPTION: Demonstrates setting Pydantic model configurations, specifically `frozen=True`, either through an internal `model_config` dictionary or directly as keyword arguments in the class definition. This affects instance immutability and enables editor-level error detection.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/integrations/visual_studio_code.md#_snippet_7

LANGUAGE: python
CODE:
```
from pydantic import BaseModel


class Knight(BaseModel):
    model_config = dict(frozen=True)
    title: str
    age: int
    color: str = 'blue'
```

LANGUAGE: python
CODE:
```
from pydantic import BaseModel


class Knight(BaseModel, frozen=True):
    title: str
    age: int
    color: str = 'blue'
```

----------------------------------------

TITLE: `missing_sentinel_error` Validation Error Example (Python)
DESCRIPTION: This error is raised when the experimental `MISSING` sentinel is the only allowed value for a field, and it is not provided during validation. The example defines a model field `f` that must be `MISSING` and attempts to validate with a different value.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_62

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ValidationError
from pydantic.experimental.missing_sentinel import MISSING


class Model(BaseModel):
    f: MISSING


try:
    Model(f=1)
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'missing_sentinel_error'
```

----------------------------------------

TITLE: `model_type` Validation Error Example (Python)
DESCRIPTION: This error is raised when the input provided to a Pydantic model is neither an instance of the model itself nor a dictionary. The example demonstrates successful validation with a dictionary and an existing model instance, followed by an error with a string input.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_64

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ValidationError


class Model(BaseModel):
    a: int
    b: int


# simply validating a dict
m = Model.model_validate({'a': 1, 'b': 2})
print(m)
#> a=1 b=2

# validating an existing model instance
print(Model.model_validate(m))
#> a=1 b=2

try:
    Model.model_validate('not an object')
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'model_type'
```

----------------------------------------

TITLE: Pydantic BaseModel with dict json_schema_extra
DESCRIPTION: Demonstrates using a dictionary with `json_schema_extra` in Pydantic's `ConfigDict` to add custom information, such as examples, to the generated JSON schema for a model.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/json_schema.md#_snippet_9

LANGUAGE: python
CODE:
```
import json

from pydantic import BaseModel, ConfigDict


class Model(BaseModel):
    a: str

    model_config = ConfigDict(json_schema_extra={'examples': [{'a': 'Foo'}]})


print(json.dumps(Model.model_json_schema(), indent=2))
```

----------------------------------------

TITLE: Model Methods and Configuration
DESCRIPTION: Details on new arguments for model methods like `.dict()` and `.json()`, and configuration options for `BaseSettings`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_294

LANGUAGE: APIDOC
CODE:
```
Model Methods:
  - Added `by_alias` argument to `.dict()` and `.json()` model methods.

BaseSettings Configuration:
  - Added `case_insensitive` option to `BaseSettings` `Config`.

Model Copying:
  - Added deep copy support for `BaseModel.copy()`.
```

----------------------------------------

TITLE: Decimal Constraints with Pydantic
DESCRIPTION: Shows how to use Pydantic's Field to constrain Decimal types, specifying the maximum number of digits and decimal places allowed. Includes a model definition and instantiation example.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/fields.md#_snippet_22

LANGUAGE: python
CODE:
```
from decimal import Decimal

from pydantic import BaseModel, Field


class Foo(BaseModel):
    precise: Decimal = Field(max_digits=5, decimal_places=2)


foo = Foo(precise=Decimal('123.45'))
print(foo)
#> precise=Decimal('123.45')
```

----------------------------------------

TITLE: Dataclass Dumping to JSON Documentation
DESCRIPTION: Adds documentation explaining how to dump Pydantic models that wrap or interact with Python dataclasses into JSON format. This guides users on serializing dataclass instances via Pydantic.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_235

LANGUAGE: python
CODE:
```
# Documentation update for serializing dataclasses with Pydantic.
# Example scenario:
# from dataclasses import dataclass
# @dataclass
# class MyData:
#     value: int
# 
# class ModelWithDataclass(BaseModel):
#     data: MyData
# 
# model = ModelWithDataclass(data=MyData(value=10))
# print(model.json()) # Demonstrates how dataclass fields are handled
```

----------------------------------------

TITLE: `multiple_of` Validation Error Example (Python)
DESCRIPTION: This error is raised when an input value does not satisfy the `multiple_of` constraint defined for a field. The example defines a model with an integer field `x` that must be a multiple of 5, and attempts to validate it with the value 1.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_66

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field, ValidationError


class Model(BaseModel):
    x: int = Field(multiple_of=5)


try:
    Model(x=1)
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'multiple_of'
```

----------------------------------------

TITLE: Pydantic init_forbid_extra Example
DESCRIPTION: Illustrates Pydantic's default behavior of ignoring extra arguments passed to a model's constructor. Setting `init_forbid_extra` to true, or configuring `extra='forbid'`, will cause an error for unexpected keyword arguments.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/integrations/mypy.md#_snippet_5

LANGUAGE: python
CODE:
```
class Model(BaseModel):
    a: int = 1


Model(unrelated=2)
```

----------------------------------------

TITLE: Debug Pydantic Models with Python-devtools
DESCRIPTION: This snippet demonstrates how to use the `debug()` function from the `python-devtools` library to inspect Pydantic `BaseModel` instances. It shows the difference in output compared to a standard `print()` statement, offering more detailed and readable debugging information. Requires `pydantic` and `python-devtools` to be installed.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/integrations/devtools.md#_snippet_0

LANGUAGE: python
CODE:
```
from datetime import datetime

from devtools import debug

from pydantic import BaseModel


class Address(BaseModel):
    street: str
    country: str
    lat: float
    lng: float


class User(BaseModel):
    id: int
    name: str
    signup_ts: datetime
    friends: list[int]
    address: Address


user = User(
    id='123',
    name='John Doe',
    signup_ts='2019-06-01 12:22',
    friends=[1234, 4567, 7890],
    address=dict(street='Testing', country='uk', lat=51.5, lng=0),
)
debug(user)
print('\nshould be much easier read than:\n')
print('user:', user)
```

----------------------------------------

TITLE: `multiple_argument_values` Validation Error Example (Python)
DESCRIPTION: This error occurs when multiple values are provided for a single argument when calling a function decorated with `validate_call`. The example defines a function `foo` that accepts argument `a` and attempts to call it with both a positional and a keyword argument for `a`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_65

LANGUAGE: python
CODE:
```
from pydantic import ValidationError, validate_call


@validate_call
def foo(a: int):
    return a


try:
    foo(1, a=2)
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'multiple_argument_values'
```

----------------------------------------

TITLE: Pydantic Private Attributes Support
DESCRIPTION: Adds support for private attributes (those starting with `__`) within Pydantic models.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_212

LANGUAGE: python
CODE:
```
from pydantic import BaseModel

class PrivateAttrModel(BaseModel):
    public_field: str
    __private_field: int = 10

    def __init__(self, **data):
        super().__init__(**data)
        # Accessing private field after initialization
        self._private_field_value = self.__private_field

# model = PrivateAttrModel(public_field='hello', __private_field=20)
# print(model.public_field)
# print(model._private_field_value) # Accessing the stored private value

```

----------------------------------------

TITLE: Pydantic Dataclass Field Validator Example
DESCRIPTION: Demonstrates using `@field_validator` in a Pydantic dataclass to transform input data. It shows how to convert an integer to a zero-padded string for a product ID field before validation.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/dataclasses.md#_snippet_8

LANGUAGE: python
CODE:
```
from pydantic import field_validator
from pydantic.dataclasses import dataclass


@dataclass
class DemoDataclass:
    product_id: str  # should be a five-digit string, may have leading zeros

    @field_validator('product_id', mode='before')
    @classmethod
    def convert_int_serial(cls, v):
        if isinstance(v, int):
            v = str(v).zfill(5)
        return v


print(DemoDataclass(product_id='01234'))
#> DemoDataclass(product_id='01234')
print(DemoDataclass(product_id=2468))
#> DemoDataclass(product_id='02468')
```

----------------------------------------

TITLE: Pydantic Valid Model Serializer Signatures
DESCRIPTION: Provides examples of valid signatures for Pydantic's model_serializer decorator, covering different modes ('plain', 'wrap') and argument inclusions (self, handler, info).

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/usage_errors.md#_snippet_40

LANGUAGE: python
CODE:
```
from pydantic import SerializerFunctionWrapHandler, SerializationInfo, model_serializer

# an instance method with the default mode or `mode='plain'`
@model_serializer  # or model_serializer(mode='plain')
def mod_ser(self, info: SerializationInfo): ...

# an instance method with `mode='wrap'`
@model_serializer(mode='wrap')
def mod_ser(self, handler: SerializerFunctionWrapHandler, info: SerializationInfo): ...

# For all of these, you can also choose to omit the `info` argument, for example:
@model_serializer(mode='plain')
def mod_ser(self): ...

@model_serializer(mode='wrap')
def mod_ser(self, handler: SerializerFunctionWrapHandler): ...
```

----------------------------------------

TITLE: Instantiate and Validate Pydantic Model
DESCRIPTION: Shows how to create an instance of a Pydantic model, including how Pydantic handles data coercion (e.g., string to integer). It highlights that instantiation performs parsing and validation, raising ValidationError on failure.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_3

LANGUAGE: python
CODE:
```
user = User(id='123')
```

----------------------------------------

TITLE: Pydantic Datetime Parsing Example
DESCRIPTION: Illustrates Pydantic's ability to parse various string and numeric formats into Python's datetime.datetime objects, including ISO 8601 with timezone information.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/standard_library_types.md#_snippet_1

LANGUAGE: python
CODE:
```
from datetime import datetime

from pydantic import BaseModel


class Event(BaseModel):
    dt: datetime = None


event = Event(dt='2032-04-23T10:20:30.400+02:30')

print(event.model_dump())
"""
{'dt': datetime.datetime(2032, 4, 23, 10, 20, 30, 400000, tzinfo=TzInfo(9000))}
"""

```

----------------------------------------

TITLE: Create Generic Pydantic Response Model
DESCRIPTION: Demonstrates creating a generic Pydantic model `Response` that can wrap data of any type. It shows usage with different data types and includes error handling for validation failures. Examples cover Python 3.9+ syntax using `typing.Generic` and Python 3.12+ syntax with new type parameter declarations.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_25

LANGUAGE: python
CODE:
```
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

DataT = TypeVar('DataT')  # (1)!


class DataModel(BaseModel):
    number: int


class Response(BaseModel, Generic[DataT]):  # (2)!
    data: DataT  # (3)!


print(Response[int](data=1))
#> data=1
print(Response[str](data='value'))
#> data='value'
print(Response[str](data='value').model_dump())
#> {'data': 'value'}

data = DataModel(number=1)
print(Response[DataModel](data=data).model_dump())
#> {'data': {'number': 1}}
try:
    Response[int](data='value')
except ValidationError as e:
    print(e)
    """
    1 validation error for Response[int]
    data
      Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='value', input_type=str]
    """

```

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ValidationError


class DataModel(BaseModel):
    number: int


class Response[DataT](BaseModel):  # (1)!
    data: DataT  # (2)!


print(Response[int](data=1))
#> data=1
print(Response[str](data='value'))
#> data='value'
print(Response[str](data='value').model_dump())
#> {'data': 'value'}

data = DataModel(number=1)
print(Response[DataModel](data=data).model_dump())
#> {'data': {'number': 1}}
try:
    Response[int](data='value')
except ValidationError as e:
    print(e)
    """
    1 validation error for Response[int]
    data
      Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='value', input_type=str]
    """

```

----------------------------------------

TITLE: `mapping_type` Validation Error Example (Python)
DESCRIPTION: This error occurs when a problem arises during validation due to a failure in a call to methods from the Mapping protocol, like `.items()`. The example demonstrates a custom Mapping class that raises an error during item access.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_57

LANGUAGE: python
CODE:
```
from collections.abc import Mapping

from pydantic import BaseModel, ValidationError


class BadMapping(Mapping):
    def items(self):
        raise ValueError()

    def __iter__(self):
        raise ValueError()

    def __getitem__(self, key):
        raise ValueError()

    def __len__(self):
        return 1


class Model(BaseModel):
    x: dict[str, str]


try:
    Model(x=BadMapping())
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'mapping_type'
```

----------------------------------------

TITLE: Pydantic type for Any Class Validation
DESCRIPTION: Demonstrates using `type` in Pydantic models to allow any class as a field value, with examples showing valid class inputs and an invalid instance input.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/standard_library_types.md#_snippet_22

LANGUAGE: Python
CODE:
```
from pydantic import BaseModel, ValidationError


class Foo:
    pass


class LenientSimpleModel(BaseModel):
    any_class_goes: type


LenientSimpleModel(any_class_goes=int)
LenientSimpleModel(any_class_goes=Foo)
try:
    LenientSimpleModel(any_class_goes=Foo())
except ValidationError as e:
    print(e)
    """
    1 validation error for LenientSimpleModel
    any_class_goes
      Input should be a type [type=is_type, input_value=<__main__.Foo object at 0x0123456789ab>, input_type=Foo]
    """

```

----------------------------------------

TITLE: Validate XML Data with Pydantic
DESCRIPTION: Demonstrates how to parse and validate data from an XML file using Python's `xml.etree.ElementTree` module and a Pydantic `BaseModel`. The example defines a `Person` model and extracts data from XML elements for validation.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/examples/files.md#_snippet_11

LANGUAGE: xml
CODE:
```
<?xml version="1.0"?>
<person>
    <name>John Doe</name>
    <age>30</age>
    <email>john@example.com</email>
</person>
```

LANGUAGE: python
CODE:
```
import xml.etree.ElementTree as ET

from pydantic import BaseModel, EmailStr, PositiveInt


class Person(BaseModel):
    name: str
    age: PositiveInt
    email: EmailStr


tree = ET.parse('person.xml').getroot()
data = {child.tag: child.text for child in tree}
person = Person.model_validate(data)
print(person)
#> name='John Doe' age=30 email='john@example.com'
```

----------------------------------------

TITLE: `model_attributes_type` Validation Error Example (Python)
DESCRIPTION: This error occurs when the input value is not a valid dictionary, model instance, or an object from which fields can be extracted. The example shows successful validation with a dictionary and a custom object, then demonstrates the error with an invalid string input.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_63

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ValidationError


class Model(BaseModel):
    a: int
    b: int


# simply validating a dict
print(Model.model_validate({'a': 1, 'b': 2}))
#> a=1 b=2


class CustomObj:
    def __init__(self, a, b):
        self.a = a
        self.b = b


# using from attributes to extract fields from an objects
print(Model.model_validate(CustomObj(3, 4), from_attributes=True))
#> a=3 b=4

try:
    Model.model_validate('not an object', from_attributes=True)
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'model_attributes_type'
```

----------------------------------------

TITLE: Pydantic Field Serializer Incorrect Signature
DESCRIPTION: Shows an unrecognized signature for Pydantic's field_serializer function. This example uses a field_serializer with no arguments, which is invalid and raises a PydanticUserError.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/usage_errors.md#_snippet_37

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, PydanticUserError, field_serializer

try:

    class Model(BaseModel):
        x: int

        @field_serializer('x')
        def no_args():
            return 'x'

except PydanticUserError as exc_info:
    assert exc_info.code == 'field-serializer-signature'
```

----------------------------------------

TITLE: Run Tests and Linting
DESCRIPTION: This command runs the full suite of Pydantic's automated tests and linting checks. It's essential to run this locally to ensure your changes haven't introduced regressions.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/contributing.md#_snippet_7

LANGUAGE: bash
CODE:
```
make
```

----------------------------------------

TITLE: Push Release Changes with Python Script
DESCRIPTION: Executes the script to push release-related changes. This action creates a pull request, applies a release label, and opens a draft release on GitHub.

SOURCE: https://github.com/pydantic/pydantic/blob/main/release/README.md#_snippet_2

LANGUAGE: shell
CODE:
```
uv run release/push.py
```

----------------------------------------

TITLE: SchemaExtraCallable for BaseConfig Type Hints
DESCRIPTION: Ensures `SchemaExtraCallable` is always defined, which helps in getting correct type hints for `BaseConfig`. This is an internal improvement for better developer experience and static analysis.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_224

LANGUAGE: python
CODE:
```
# Internal Pydantic change related to schema generation and type hinting.
# Affects how BaseConfig interacts with schema generation functions.
```

----------------------------------------

TITLE: Handling Deprecation Warnings in Validators
DESCRIPTION: Provides an example of how to access a deprecated field within a Pydantic validator while suppressing the deprecation warning using `warnings.catch_warnings` and `warnings.simplefilter`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/fields.md#_snippet_33

LANGUAGE: Python
CODE:
```
import warnings

from typing_extensions import Self

from pydantic import BaseModel, Field, model_validator


class Model(BaseModel):
    deprecated_field: int = Field(deprecated='This is deprecated')

    @model_validator(mode='after')
    def validate_model(self) -> Self:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            self.deprecated_field = self.deprecated_field * 2

```

----------------------------------------

TITLE: Pydantic V2: Mypy Plugin Configuration
DESCRIPTION: Shows how to configure Mypy to use Pydantic V2's Mypy plugin, and optionally the V1 plugin if needed for compatibility. Configuration is provided for both mypy.ini and pyproject.toml.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_33

LANGUAGE: ini
CODE:
```
[mypy]
plugins = pydantic.mypy, pydantic.v1.mypy  # include `.v1.mypy` if required.
```

----------------------------------------

TITLE: JSON Mode Serialization with model_dump_json()
DESCRIPTION: Shows how to serialize Pydantic models directly to a JSON-encoded string using `model_dump_json()`. Includes an example with pretty-printing using `indent=2` and highlights Pydantic's support for various data types.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/serialization.md#_snippet_1

LANGUAGE: python
CODE:
```
from datetime import datetime

from pydantic import BaseModel


class BarModel(BaseModel):
    whatever: tuple[int, ...]


class FooBarModel(BaseModel):
    foo: datetime
    bar: BarModel


m = FooBarModel(foo=datetime(2032, 6, 1, 12, 13, 14), bar={'whatever': (1, 2)})

print(m.model_dump_json(indent=2))
```

----------------------------------------

TITLE: JSON Schema for Person Data
DESCRIPTION: An example JSON Schema defining the structure for a 'Person' object. It includes properties like first name, last name, age, pets, and a comment, along with type definitions and required fields.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/integrations/datamodel_code_generator.md#_snippet_2

LANGUAGE: json
CODE:
```
{
  "$id": "person.json",
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Person",
  "type": "object",
  "properties": {
    "first_name": {
      "type": "string",
      "description": "The person's first name."
    },
    "last_name": {
      "type": "string",
      "description": "The person's last name."
    },
    "age": {
      "description": "Age in years.",
      "type": "integer",
      "minimum": 0
    },
    "pets": {
      "type": "array",
      "items": [
        {
          "$ref": "#/definitions/Pet"
        }
      ]
    },
    "comment": {
      "type": "null"
    }
  },
  "required": [
      "first_name",
      "last_name"
  ],
  "definitions": {
    "Pet": {
      "properties": {
        "name": {
          "type": "string"
        },
        "age": {
          "type": "integer"
        }
      }
    }
  }
}
```

----------------------------------------

TITLE: Configuration for Standard Library Dataclasses
DESCRIPTION: Shows how to configure standard library dataclasses using the `__pydantic_config__` class attribute with `ConfigDict`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/config.md#_snippet_4

LANGUAGE: python
CODE:
```
from dataclasses import dataclass

from pydantic import ConfigDict


@dataclass
class User:
    __pydantic_config__ = ConfigDict(strict=True)

    id: int
    name: str = 'John Doe'

```

----------------------------------------

TITLE: Handle bytes_invalid_encoding Validation Error in Pydantic
DESCRIPTION: Details the 'bytes_invalid_encoding' error when bytes data is invalid under the configured encoding, using hex encoding as an example.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_4

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ValidationError


class Model(BaseModel):
    x: bytes
    model_config = {'val_json_bytes': 'hex'}


try:
    Model(x=b'a')
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'bytes_invalid_encoding'
```

----------------------------------------

TITLE: Validate TOML Data with Pydantic
DESCRIPTION: Shows how to parse and validate data from a TOML configuration file using Python's `tomllib` module and a Pydantic `BaseModel`. The example defines a `Person` model and validates the loaded TOML data.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/examples/files.md#_snippet_9

LANGUAGE: toml
CODE:
```
name = "John Doe"
age = 30
email = "john@example.com"
```

LANGUAGE: python
CODE:
```
import tomllib

from pydantic import BaseModel, EmailStr, PositiveInt


class Person(BaseModel):
    name: str
    age: PositiveInt
    email: EmailStr


with open('person.toml', 'rb') as f:
    data = tomllib.load(f)

person = Person.model_validate(data)
print(person)
#> name='John Doe' age=30 email='john@example.com'
```

----------------------------------------

TITLE: Pydantic Valid Field Serializer Signatures
DESCRIPTION: Provides examples of valid signatures for Pydantic's field_serializer decorator, covering different modes ('plain', 'wrap') and argument inclusions (self, value, handler, info).

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/usage_errors.md#_snippet_38

LANGUAGE: python
CODE:
```
from pydantic import FieldSerializationInfo, SerializerFunctionWrapHandler, field_serializer

# an instance method with the default mode or `mode='plain'`
@field_serializer('x')  # or @field_serializer('x', mode='plain')
def ser_x(self, value: Any, info: FieldSerializationInfo): ...

# a static method or function with the default mode or `mode='plain'`
@field_serializer('x')  # or @field_serializer('x', mode='plain')
@staticmethod
def ser_x(value: Any, info: FieldSerializationInfo): ...

# equivalent to
def ser_x(value: Any, info: FieldSerializationInfo): ...
serializer('x')(ser_x)

# an instance method with `mode='wrap'`
@field_serializer('x', mode='wrap')
def ser_x(self, value: Any, nxt: SerializerFunctionWrapHandler, info: FieldSerializationInfo): ...

# a static method or function with `mode='wrap'`
@field_serializer('x', mode='wrap')
@staticmethod
def ser_x(value: Any, nxt: SerializerFunctionWrapHandler, info: FieldSerializationInfo): ...

# equivalent to
def ser_x(value: Any, nxt: SerializerFunctionWrapHandler, info: FieldSerializationInfo): ...
serializer('x')(ser_x)

# For all of these, you can also choose to omit the `info` argument, for example:
@field_serializer('x')
def ser_x(self, value: Any): ...

@field_serializer('x', mode='wrap')
def ser_x(self, value: Any, handler: SerializerFunctionWrapHandler): ...
```

----------------------------------------

TITLE: Model Field Missing Annotation (ClassVar): Python Example
DESCRIPTION: Demonstrates resolving the 'model-field-missing-annotation' error by annotating a field as a ClassVar, indicating it's not intended to be a model field.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/usage_errors.md#_snippet_16

LANGUAGE: python
CODE:
```
from typing import ClassVar

from pydantic import BaseModel


class Model(BaseModel):
    a: ClassVar[str]
```

----------------------------------------

TITLE: Pydantic Dataclass Model Validators and Post-Initialization Hooks
DESCRIPTION: Illustrates the execution order of `@model_validator` (both `before` and `after` modes) and the `__post_init__` method in a Pydantic dataclass, showing how `ArgsKwargs` is used for `before` validators.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/dataclasses.md#_snippet_10

LANGUAGE: python
CODE:
```
from pydantic_core import ArgsKwargs
from typing_extensions import Self

from pydantic import model_validator
from pydantic.dataclasses import dataclass


@dataclass
class Birth:
    year: int
    month: int
    day: int


@dataclass
class User:
    birth: Birth

    @model_validator(mode='before')
    @classmethod
    def before(cls, values: ArgsKwargs) -> ArgsKwargs:
        print(f'First: {values}')  # (1)!
        """
        First: ArgsKwargs((), {'birth': {'year': 1995, 'month': 3, 'day': 2}})
        """
        return values

    @model_validator(mode='after')
    def after(self) -> Self:
        print(f'Third: {self}')
        # Third: User(birth=Birth(year=1995, month=3, day=2))
        return self

    def __post_init__(self):
        print(f'Second: {self.birth}')
        # Second: Birth(year=1995, month=3, day=2)


user = User(**{'birth': {'year': 1995, 'month': 3, 'day': 2}})
```

----------------------------------------

TITLE: Performance: Improve Annotation Application
DESCRIPTION: Boosts the performance of applying type annotations within Pydantic models. This optimization contributes to faster model definition and validation setup.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_33

LANGUAGE: python
CODE:
```
# Internal improvements to how type annotations are processed and applied.
```

----------------------------------------

TITLE: Serialization of Date Subclasses
DESCRIPTION: Illustrates how Pydantic serializes subclasses of standard types like `datetime.date`. The example shows that the subclass's custom properties are not included in the default serialization.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/serialization.md#_snippet_15

LANGUAGE: python
CODE:
```
from datetime import date

from pydantic import BaseModel


class MyDate(date):
    @property
    def my_date_format(self) -> str:
        return self.strftime('%d/%m/%Y')


class FooModel(BaseModel):
    date: date


m = FooModel(date=MyDate(2023, 1, 1))
print(m.model_dump_json())
```

----------------------------------------

TITLE: Excluding Fields During Export
DESCRIPTION: Explains how to use the `exclude=True` parameter on a Pydantic `Field` to prevent that field from being included when exporting the model, for example, using `model_dump()`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/fields.md#_snippet_29

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field


class User(BaseModel):
    name: str
    age: int = Field(exclude=True)


user = User(name='John', age=42)
print(user.model_dump())  # (1)!
#> {'name': 'John'}
```

----------------------------------------

TITLE: Parametrize Generic Model with `int`
DESCRIPTION: Shows how to instantiate a generic Pydantic model (`ChildClass`) by providing concrete types for its type variables. This example specifically parameterizes `TypeX` with `int` and demonstrates the resulting object's attribute.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_27

LANGUAGE: python
CODE:
```
print(ChildClass[int](X=1))
#> X=1
```

----------------------------------------

TITLE: Settings Priority Documentation
DESCRIPTION: Improved documentation regarding settings priority and provided mechanisms to easily change it, enhancing flexibility in configuration management.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_279

LANGUAGE: python
CODE:
```
from pydantic import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):
    database_url: str

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8'
    )
```

----------------------------------------

TITLE: Customize Validation with __get_pydantic_core_schema__
DESCRIPTION: Shows how to customize Pydantic validation for custom types by implementing the `__get_pydantic_core_schema__` class method. This example creates a `Username` type inheriting from `str` and customizes its core schema generation.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/types.md#_snippet_12

LANGUAGE: Python
CODE:
```
from typing import Any

from pydantic_core import CoreSchema, core_schema

from pydantic import GetCoreSchemaHandler, TypeAdapter


class Username(str):
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(cls, handler(str))


ta = TypeAdapter(Username)
res = ta.validate_python('abc')
assert isinstance(res, Username)
assert res == 'abc'

```

----------------------------------------

TITLE: Model Field Missing Annotation (ignored_types): Python Example
DESCRIPTION: Shows how to resolve the 'model-field-missing-annotation' error by configuring Pydantic to ignore specific types, preventing them from being treated as model fields.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/usage_errors.md#_snippet_17

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ConfigDict


class IgnoredType:
    pass


class MyModel(BaseModel):
    model_config = ConfigDict(ignored_types=(IgnoredType,))

    _a = IgnoredType()
    _b: int = IgnoredType()
    _c: IgnoredType
    _d: IgnoredType = IgnoredType()
```

----------------------------------------

TITLE: Pydantic Arguments Schema API
DESCRIPTION: Documentation for the experimental Arguments Schema API in Pydantic. This API allows for the generation of schemas that describe function arguments.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/experimental.md#_snippet_1

LANGUAGE: APIDOC
CODE:
```
pydantic.experimental.arguments_schema.generate_arguments_schema:
  Generates a Pydantic schema from a function's signature.
  
  Purpose:
  This function inspects a given Python function and creates a Pydantic model that represents the function's arguments, including their types, defaults, and validation rules.
  
  Parameters:
  - func: The Python function for which to generate the arguments schema.
  - **kwargs: Additional keyword arguments to pass to the schema generation process.
  
  Returns:
  A Pydantic model class representing the function's arguments.
  
  Example:
  from typing import Annotated
  
  def greet(name: str, age: Annotated[int, 'Age in years'] = 30):
      print(f"Hello {name}, you are {age} years old.")
  
  ArgumentsSchema = generate_arguments_schema(greet)
  
  # Now ArgumentsSchema can be used to validate input for the greet function:
  valid_args = ArgumentsSchema(name='Alice', age=25)
  invalid_args = ArgumentsSchema(name='Bob', age='twenty') # This would raise a ValidationError
  
  # Note: Specific parameter details and return types are inferred from the source code.
```

----------------------------------------

TITLE: Import Pydantic V1 Fields Module (>=1.10.17)
DESCRIPTION: Illustrates importing a module like 'fields' from the Pydantic V1 namespace, specifically for Pydantic versions 1.10.17 and later. This import pattern works within both V1 and V2 installations.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_6

LANGUAGE: python
CODE:
```
from pydantic.v1.fields import ModelField
```

----------------------------------------

TITLE: API: pydantic.main.create_model
DESCRIPTION: Provides documentation for the `create_model` function, which allows dynamic creation of Pydantic models at runtime. Explains how to define fields with types and default values.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_41

LANGUAGE: APIDOC
CODE:
```
pydantic.main.create_model

Create a model using runtime information to specify the fields.

Parameters:
  - __main__: The name of the model to create.
  - **field_name: type, default_value, validator, etc.**
    Fields can be defined as keyword arguments. The value can be a type, a tuple of (type, default_value), or a more complex configuration including validators.

Returns:
  - A new Pydantic BaseModel class.

Example:
```python
from pydantic import BaseModel, create_model

DynamicFoobarModel = create_model('DynamicFoobarModel', foo=str, bar=(int, 123))

# Usage:
instance = DynamicFoobarModel(foo='hello')
print(instance.foo) # Output: hello
print(instance.bar) # Output: 123
```

Related:
  - `pydantic.create_model` (alias for `pydantic.main.create_model`)
```

----------------------------------------

TITLE: Pydantic Model Serializer Incorrect Signature
DESCRIPTION: Illustrates an unrecognized signature for Pydantic's model_serializer function. This example shows a model_serializer with an incorrect number of arguments, leading to a PydanticUserError.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/usage_errors.md#_snippet_39

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, PydanticUserError, model_serializer

try:

    class MyModel(BaseModel):
        a: int

        @model_serializer
        def _serialize(self, x, y, z):
            return self

except PydanticUserError as exc_info:
    assert exc_info.code == 'model-serializer-signature'
```

----------------------------------------

TITLE: Pydantic Badges in HTML
DESCRIPTION: Illustrates how to integrate Pydantic version badges into HTML content. The provided HTML snippets create clickable badges that link to the Pydantic website.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/contributing.md#_snippet_14

LANGUAGE: html
CODE:
```
<a href="https://pydantic.dev"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v1.json" alt="Pydantic Version 1" style="max-width:100%;"></a>

<a href="https://pydantic.dev"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json" alt="Pydantic Version 2" style="max-width:100%;"></a>
```

----------------------------------------

TITLE: mkdocstrings Cross-Reference for Pydantic API
DESCRIPTION: Configure mkdocstrings within MkDocs to link to Pydantic's API documentation. This is achieved by specifying the Pydantic object inventory URL in the mkdocstrings plugin configuration in your `mkdocs.yml` file.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/integrations/documentation.md#_snippet_1

LANGUAGE: YAML
CODE:
```
plugins:
- mkdocstrings:
    handlers:
      python:
        import:
        - https://docs.pydantic.dev/latest/objects.inv
```

----------------------------------------

TITLE: Pydantic Literal Validation
DESCRIPTION: Demonstrates using typing.Literal to enforce specific string values for Pydantic model fields. Includes examples of valid inputs and how validation errors are raised for invalid inputs.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/standard_library_types.md#_snippet_25

LANGUAGE: python
CODE:
```
from typing import Literal

from pydantic import BaseModel, ValidationError


class Pie(BaseModel):
    flavor: Literal['apple', 'pumpkin']


Pie(flavor='apple')
Pie(flavor='pumpkin')
try:
    Pie(flavor='cherry')
except ValidationError as e:
    print(str(e))
    
```

----------------------------------------

TITLE: Pydantic validate_call Decorator Usage
DESCRIPTION: Demonstrates how to use the @validate_call decorator to validate function arguments against type annotations. Includes examples of correct usage, type coercion, and handling validation errors.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/validation_decorator.md#_snippet_0

LANGUAGE: python
CODE:
```
from pydantic import ValidationError, validate_call


@validate_call
def repeat(s: str, count: int, *, separator: bytes = b'') -> bytes:
    b = s.encode()
    return separator.join(b for _ in range(count))


a = repeat('hello', 3)
print(a)
#> b'hellohellohello'

b = repeat('x', '4', separator=b' ')
print(b)
#> b'x x x x'

try:
    c = repeat('hello', 'wrong')
except ValidationError as exc:
    print(exc)
    """
    1 validation error for repeat
    1
      Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='wrong', input_type=str]
    """

```

LANGUAGE: python
CODE:
```
from datetime import date

from pydantic import validate_call


@validate_call
def greater_than(d1: date, d2: date, *, include_equal=False) -> date:  # (1)!
    if include_equal:
        return d1 >= d2
    else:
        return d1 > d2


d1 = '2000-01-01'  # (2)!
d2 = date(2001, 1, 1)
greater_than(d1, d2, include_equal=True)

```

----------------------------------------

TITLE: Pydantic Dataclass Model Validator and Post-Initialization Hooks
DESCRIPTION: Illustrates the execution order of `@model_validator` (before/after) and the `__post_init__` method in Pydantic dataclasses. It shows how data flows through these hooks during initialization, with a note on the `ArgsKwargs` type for 'before' validators.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/dataclasses.md#_snippet_9

LANGUAGE: python
CODE:
```
from pydantic_core import ArgsKwargs
from typing_extensions import Self

from pydantic import model_validator
from pydantic.dataclasses import dataclass


@dataclass
class Birth:
    year: int
    month: int
    day: int


@dataclass
class User:
    birth: Birth

    @model_validator(mode='before')
    @classmethod
    def before(cls, values: ArgsKwargs) -> ArgsKwargs:
        print(f'First: {values}')  # (1)!
        """
        First: ArgsKwargs((), {'birth': {'year': 1995, 'month': 3, 'day': 2}})
        """
        return values

    @model_validator(mode='after')
    def after(self) -> Self:
        print(f'Third: {self}')
        #> Third: User(birth=Birth(year=1995, month=3, day=2))
        return self

    def __post_init__(self):
        print(f'Second: {self.birth}')
        #> Second: Birth(year=1995, month=3, day=2)


user = User(**{'birth': {'year': 1995, 'month': 3, 'day': 2}})

# 1. Unlike Pydantic models, the `values` parameter is of type [`ArgsKwargs`][pydantic_core.ArgsKwargs]
```

----------------------------------------

TITLE: Field Representation Control with Pydantic
DESCRIPTION: Explains how to use the `repr` parameter in Pydantic's Field to include or exclude fields from the model's string representation. Provides an example demonstrating this functionality.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/fields.md#_snippet_24

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field


class User(BaseModel):
    name: str = Field(repr=True)  # (1)!
    age: int = Field(repr=False)


user = User(name='John', age=42)
print(user)
#> name='John'
```

----------------------------------------

TITLE: Pydantic Pipeline API
DESCRIPTION: Documentation for the experimental Pipeline API in Pydantic. This API is designed for building and managing data processing pipelines.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/experimental.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
pydantic.experimental.pipeline._Pipeline:
  A class for defining and executing data processing pipelines.
  
  Usage:
  Define a pipeline by subclassing _Pipeline and implementing the steps.
  
  Example:
  class MyPipeline(_Pipeline):
      def step1(self, data):
          # Process data
          return processed_data
      
      def step2(self, data):
          # Further process data
          return final_data
  
  pipeline = MyPipeline()
  result = pipeline.run(initial_data)
  
  # Note: Specific methods, parameters, and return types are inferred from the source code.
```

----------------------------------------

TITLE: Pydantic V2: Mypy Plugin Configuration
DESCRIPTION: Shows how to configure Mypy to use Pydantic V2's Mypy plugin, and optionally the V1 plugin if needed for compatibility. Configuration is provided for both mypy.ini and pyproject.toml.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_34

LANGUAGE: toml
CODE:
```
[tool.mypy]
plugins = [
    "pydantic.mypy",
    "pydantic.v1.mypy",  # include `.v1.mypy` if required.
]
```

----------------------------------------

TITLE: Pydantic Pipeline API Usage
DESCRIPTION: Illustrates the experimental 'pipeline' API for composing validation, constraints, and transformations in a type-safe manner. It shows how to use `validate_as` with various methods like `str_lower`, `gt`, `str_pattern`, `transform`, `predicate`, and combining steps with `|` or `&`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/experimental.md#_snippet_1

LANGUAGE: python
CODE:
```
from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel
from pydantic.experimental.pipeline import validate_as


class User(BaseModel):
    name: Annotated[str, validate_as(str).str_lower()]  # (1)!
    age: Annotated[int, validate_as(int).gt(0)]  # (2)!
    username: Annotated[str, validate_as(str).str_pattern(r'[a-z]+')]  # (3)!
    password:
        Annotated[
            str,
            validate_as(str)
            .transform(str.lower)
            .predicate(lambda x: x != 'password'),  # (4)!
        ]
    favorite_number: Annotated[
        int,
        (validate_as(int) | validate_as(str).str_strip().validate_as(int)).gt(
            0
        ),
    ]  # (5)!
    friends: Annotated[list[User], validate_as(...).len(0, 100)]  # (6)!
    bio:
        Annotated[
            datetime,
            validate_as(int)
            .transform(lambda x: x / 1_000_000)
            .validate_as(...),  # (8)!
        ]
```

----------------------------------------

TITLE: datetime_object_invalid Pydantic Validation Example
DESCRIPTION: Demonstrates the 'datetime_object_invalid' error, raised when a datetime object has an invalid tzinfo implementation. It shows how to define a custom tzinfo and trigger this validation error.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_21

LANGUAGE: python
CODE:
```
from datetime import datetime, tzinfo

from pydantic import AwareDatetime, BaseModel, ValidationError


class CustomTz(tzinfo):
    # utcoffset is not implemented!

    def tzname(self, _dt):
        return 'CustomTZ'


class Model(BaseModel):
    x: AwareDatetime


try:
    Model(x=datetime(2023, 1, 1, tzinfo=CustomTz()))
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'datetime_object_invalid'
```

----------------------------------------

TITLE: Demonstrate `needs_python_object` Pydantic Validation Error in Python
DESCRIPTION: This error is raised when validation is attempted from a format that cannot be converted to a Python object. For example, attempting to validate a string as a BaseModel class.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_67

LANGUAGE: python
CODE:
```
import json

from pydantic import BaseModel, ValidationError


class Model(BaseModel):
    bm: type[BaseModel]


try:
    Model.model_validate_json(json.dumps({'bm': 'not a basemodel class'}))
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'needs_python_object'
```

----------------------------------------

TITLE: Pydantic type[T] Subclass Validation
DESCRIPTION: Illustrates using `type[T]` in Pydantic models to enforce that a field must be a subclass of a specified type `T`, with examples of valid and invalid inputs.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/standard_library_types.md#_snippet_21

LANGUAGE: Python
CODE:
```
from pydantic import BaseModel, ValidationError


class Foo:
    pass


class Bar(Foo):
    pass


class Other:
    pass


class SimpleModel(BaseModel):
    just_subclasses: type[Foo]


SimpleModel(just_subclasses=Foo)
SimpleModel(just_subclasses=Bar)
try:
    SimpleModel(just_subclasses=Other)
except ValidationError as e:
    print(e)
    """
    1 validation error for SimpleModel
    just_subclasses
      Input should be a subclass of Foo [type=is_subclass_of, input_value=<class '__main__.Other'>, input_type=type]
    """

```

----------------------------------------

TITLE: Pydantic Decimal Validation and Serialization
DESCRIPTION: Demonstrates Pydantic's handling of the `decimal.Decimal` type. It explains validation by converting to string and then to Decimal, and serialization to strings. Includes an example using `PlainSerializer` to override default serialization behavior, allowing `Decimal` to be serialized as a float.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/standard_library_types.md#_snippet_6

LANGUAGE: python
CODE:
```
from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, PlainSerializer


class Model(BaseModel):
    x: Decimal
    y: Annotated[
        Decimal,
        PlainSerializer(
            lambda x: float(x), return_type=float, when_used='json'
        ),
    ]


my_model = Model(x=Decimal('1.1'), y=Decimal('2.1'))

print(my_model.model_dump())
print(my_model.model_dump(mode='json'))
print(my_model.model_dump_json())
```

----------------------------------------

TITLE: Pydantic Field Validator Incorrect Signature
DESCRIPTION: Demonstrates an unrecognized signature for Pydantic's field_validator or model_validator functions. This example shows a field_validator with an incorrect signature, leading to a PydanticUserError.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/usage_errors.md#_snippet_36

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, PydanticUserError, field_validator

try:

    class Model(BaseModel):
        a: str

        @field_validator('a')
        @classmethod
        def check_a(cls):
            return 'a'

except PydanticUserError as exc_info:
    assert exc_info.code == 'validator-signature'
```

----------------------------------------

TITLE: Pydantic V1 vs V2 JSON Key Serialization
DESCRIPTION: Compares how Pydantic V1 and V2 serialize dictionary keys, particularly `None` values, to JSON. It demonstrates the output differences between V1's `.json()` and V2's `.model_dump_json()` methods.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_15

LANGUAGE: python
CODE:
```
from typing import Optional

from pydantic import BaseModel as V2BaseModel
from pydantic.v1 import BaseModel as V1BaseModel


class V1Model(V1BaseModel):
    a: dict[Optional[str], int]


class V2Model(V2BaseModel):
    a: dict[Optional[str], int]


v1_model = V1Model(a={None: 123})
v2_model = V2Model(a={None: 123})

# V1
print(v1_model.json())
#> {"a": {"null": 123}}

# V2
print(v2_model.model_dump_json())
#> {"a":{"None":123}}
```

----------------------------------------

TITLE: Pydantic V1 Validator Incorrect Signature
DESCRIPTION: Illustrates an unsupported signature for Pydantic V1-style validators, which raises a PydanticUserError. This example shows how an incorrect parameter in a validator method can cause issues.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/usage_errors.md#_snippet_35

LANGUAGE: python
CODE:
```
import warnings

from pydantic import BaseModel, PydanticUserError, validator

warnings.filterwarnings('ignore', category=DeprecationWarning)

try:

    class Model(BaseModel):
        a: int

        @validator('a')
        def check_a(cls, value, foo):
            return value

except PydanticUserError as exc_info:
    assert exc_info.code == 'validator-v1-signature'
```

----------------------------------------

TITLE: Pydantic BaseModel with Callable json_schema_extra
DESCRIPTION: Illustrates how to use a callable function with `json_schema_extra` to dynamically modify the JSON schema. This example shows a function that removes the 'default' key from the schema.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/json_schema.md#_snippet_10

LANGUAGE: python
CODE:
```
import json

from pydantic import BaseModel, Field


def pop_default(s):
    s.pop('default')


class Model(BaseModel):
    a: int = Field(default=1, json_schema_extra=pop_default)


print(json.dumps(Model.model_json_schema(), indent=2))
```

----------------------------------------

TITLE: Pydantic Field Customization with Field Parameters
DESCRIPTION: Demonstrates using Pydantic's Field function to customize model fields with parameters like description, examples, title, and json_schema_extra. This allows for richer metadata and control over the generated JSON schema.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/json_schema.md#_snippet_5

LANGUAGE: python
CODE:
```
import json

from pydantic import BaseModel, EmailStr, Field, SecretStr


class User(BaseModel):
    age: int = Field(description='Age of the user')
    email: EmailStr = Field(examples=['marcelo@mail.com'])
    name: str = Field(title='Username')
    password: SecretStr = Field(
        json_schema_extra={
            'title': 'Password',
            'description': 'Password of the user',
            'examples': ['123456'],
        }
    )


print(json.dumps(User.model_json_schema(), indent=2))
```

LANGUAGE: json
CODE:
```
{
  "properties": {
    "age": {
      "description": "Age of the user",
      "title": "Age",
      "type": "integer"
    },
    "email": {
      "examples": [
        "marcelo@mail.com"
      ],
      "format": "email",
      "title": "Email",
      "type": "string"
    },
    "name": {
      "title": "Username",
      "type": "string"
    },
    "password": {
      "description": "Password of the user",
      "examples": [
        "123456"
      ],
      "format": "password",
      "title": "Password",
      "type": "string",
      "writeOnly": true
    }
  },
  "required": [
    "age",
    "email",
    "name",
    "password"
  ],
  "title": "User",
  "type": "object"
}
```

----------------------------------------

TITLE: Pydantic TypeAdapter with TypedDict Schema
DESCRIPTION: Demonstrates using TypeAdapter to create a schema from a TypedDict, including validation of datetime and bytes, serialization, and JSON schema generation. It shows how to handle different input types and output formats.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/why.md#_snippet_5

LANGUAGE: python
CODE:
```
from datetime import datetime

from typing_extensions import NotRequired, TypedDict

from pydantic import TypeAdapter


class Meeting(TypedDict):
    when: datetime
    where: bytes
    why: NotRequired[str]


meeting_adapter = TypeAdapter(Meeting)
m = meeting_adapter.validate_python(
    {'when': '2020-01-01T12:00', 'where': 'home'}
) # (1)!
print(m)
#> {'when': datetime.datetime(2020, 1, 1, 12, 0), 'where': b'home'}
meeting_adapter.dump_python(m, exclude={'where'})  # (2)!

print(meeting_adapter.json_schema())  # (3)!
"""
{
    'properties': {
        'when': {'format': 'date-time', 'title': 'When', 'type': 'string'},
        'where': {'format': 'binary', 'title': 'Where', 'type': 'string'},
        'why': {'title': 'Why', 'type': 'string'},
    },
    'required': ['when', 'where'],
    'title': 'Meeting',
    'type': 'object',
}
"""

```

----------------------------------------

TITLE: Discriminator No Field: Union Example
DESCRIPTION: Illustrates the 'discriminator-no-field' error, which arises when a model within a discriminated union (using `typing.Union` and `pydantic.Field(discriminator=...)`) does not define the specified discriminator field.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/usage_errors.md#_snippet_7

LANGUAGE: python
CODE:
```
from typing import Literal, Union

from pydantic import BaseModel, Field, PydanticUserError


class Cat(BaseModel):
    c: str


class Dog(BaseModel):
    pet_type: Literal['dog']
    d: str


try:

    class Model(BaseModel):
        pet: Union[Cat, Dog] = Field(discriminator='pet_type')
        number: int

except PydanticUserError as exc_info:
    assert exc_info.code == 'discriminator-no-field'
```

----------------------------------------

TITLE: BaseModel Methods and Properties
DESCRIPTION: Reference for core methods and attributes of Pydantic's BaseModel, covering data validation, serialization, model construction, copying, schema generation, and field introspection.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_8

LANGUAGE: APIDOC
CODE:
```
Pydantic BaseModel Methods and Properties:

model_validate(obj: Any)
  - Validates the given object against the Pydantic model.

model_validate_json(json_data: str)
  - Validates the given JSON data against the Pydantic model.

model_construct(**values: Any)
  - Creates models without running validation.

model_dump(**kwargs: Any)
  - Returns a dictionary of the model's fields and values.

model_dump_json(**kwargs: Any)
  - Returns a JSON string representation of model_dump().

model_copy(**kwargs: Any)
  - Returns a copy (by default, shallow copy) of the model.

model_json_schema(**kwargs: Any)
  - Returns a jsonable dictionary representing the model's JSON Schema.

model_fields
  - A mapping between field names and their definitions.

model_computed_fields
  - A mapping between computed field names and their definitions.

model_extra
  - The extra fields set during validation.

model_fields_set
  - The set of fields which were explicitly provided when the model was initialized.

model_parametrized_name()
  - Computes the class name for parametrizations of generic classes.

model_post_init()
  - Performs additional actions after the model is instantiated and all field validators are applied.

model_rebuild()
  - Rebuilds the model schema, supporting recursive generic models.
```

----------------------------------------

TITLE: Pydantic PlainValidator with Annotated Pattern
DESCRIPTION: Illustrates using `PlainValidator` with `Annotated` to modify input values directly, bypassing subsequent Pydantic validation. This example shows doubling an integer input, and how non-integer inputs are accepted without further type checking.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/validators.md#_snippet_6

LANGUAGE: python
CODE:
```
from typing import Annotated, Any

from pydantic import BaseModel, PlainValidator


def val_number(value: Any) -> Any:
    if isinstance(value, int):
        return value * 2
    else:
        return value


class Model(BaseModel):
    number: Annotated[int, PlainValidator(val_number)]


print(Model(number=4))
#> number=8
print(Model(number='invalid'))  # (1)!
#> number='invalid'

```

----------------------------------------

TITLE: Python: Validate Unparametrized Type Variables
DESCRIPTION: Explains Pydantic's behavior when type variables are left unparametrized, falling back to `Any` or using bounds/defaults. Shows validation examples with `TypeVar` constraints and default values.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_35

LANGUAGE: python
CODE:
```
from typing import Generic

from typing_extensions import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar('T')
U = TypeVar('U', bound=int)
V = TypeVar('V', default=str)


class Model(BaseModel, Generic[T, U, V]):
    t: T
    u: U
    v: V


print(Model(t='t', u=1, v='v'))

try:
    Model(t='t', u='u', v=1)
except ValidationError as exc:
    print(exc)
```

----------------------------------------

TITLE: Serialize Dictionary with Cyclic Reference (Pydantic)
DESCRIPTION: Demonstrates Pydantic's default behavior when serializing a Python dictionary containing a cyclic reference. It shows the initial setup of cyclic data and the resulting ValueError raised by TypeAdapter during serialization.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/forward_annotations.md#_snippet_5

LANGUAGE: python
CODE:
```
from pydantic import TypeAdapter

# Create data with cyclic references representing the graph 1 -> 2 -> 3 -> 1
node_data = {'id': 1, 'children': [{'id': 2, 'children': [{'id': 3}]}]}
node_data['children'][0]['children'][0]['children'] = [node_data]

try:
    # Try serializing the circular reference as JSON
    TypeAdapter(dict).dump_json(node_data)
except ValueError as exc:
    print(exc)
    """
    Error serializing to JSON: ValueError: Circular reference detected (id repeated)
    """

```

----------------------------------------

TITLE: ConfigDict(strict=True) for BaseModel
DESCRIPTION: Illustrates enabling strict mode globally for a `BaseModel` by setting `model_config = ConfigDict(strict=True)`. This example shows how inputs like string '33' for an integer field and 'yes' for a boolean field will raise `ValidationError`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/strict_mode.md#_snippet_13

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ConfigDict, ValidationError


class User(BaseModel):
    model_config = ConfigDict(strict=True)

    name: str
    age: int
    is_active: bool


try:
    User(name='David', age='33', is_active='yes')
except ValidationError as exc:
    print(exc)
    
```

----------------------------------------

TITLE: Custom Configuration for validate_call with arbitrary_types_allowed
DESCRIPTION: Illustrates how to apply custom configurations to the @validate_call decorator using Pydantic's ConfigDict. This example enables arbitrary types, allowing custom classes like 'Foobar' to be passed as arguments.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/validation_decorator.md#_snippet_14

LANGUAGE: Python
CODE:
```
from pydantic import ConfigDict, ValidationError, validate_call


class Foobar:
    def __init__(self, v: str):
        self.v = v

    def __add__(self, other: 'Foobar') -> str:
        return f'{self} + {other}'

    def __str__(self) -> str:
        return f'Foobar({self.v})'


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def add_foobars(a: Foobar, b: Foobar):
    return a + b


c = add_foobars(Foobar('a'), Foobar('b'))
print(c)
# > Foobar(a) + Foobar(b)

try:
    add_foobars(1, 2)
except ValidationError as e:
    print(e)
    # 2 validation errors for add_foobars
    # 0
    #   Input should be an instance of Foobar [type=is_instance_of, input_value=1, input_type=int]
    # 1
    #   Input should be an instance of Foobar [type=is_instance_of, input_value=2, input_type=int]
    # 
```

----------------------------------------

TITLE: Custom Validation with Annotated and GetPydanticSchema
DESCRIPTION: Shows how to reduce boilerplate for custom type validation using `Annotated` and `GetPydanticSchema`. This example defines a validator that doubles a string input, demonstrating a concise way to apply custom logic.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/types.md#_snippet_16

LANGUAGE: python
CODE:
```
from typing import Annotated

from pydantic import BaseModel, GetPydanticSchema
from pydantic_core import core_schema


class Model(BaseModel):
    y: Annotated[
        str,
        GetPydanticSchema(
            lambda tp, handler: core_schema.no_info_after_validator_function(
                lambda x: x * 2, handler(tp)
            )
        ),
    ]


assert Model(y='ab').y == 'abab'
```

----------------------------------------

TITLE: Handling Invalid Schemas with Custom Generator
DESCRIPTION: Illustrates how to exclude fields from JSON schema generation that are invalid by overriding `handle_invalid_for_json_schema` to raise `PydanticOmit`. Includes an example with a `Callable` field.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/json_schema.md#_snippet_23

LANGUAGE: python
CODE:
```
from typing import Callable

from pydantic_core import PydanticOmit, core_schema

from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue


class MyGenerateJsonSchema(GenerateJsonSchema):
    def handle_invalid_for_json_schema(
        self, schema: core_schema.CoreSchema, error_info: str
    ) -> JsonSchemaValue:
        raise PydanticOmit


def example_callable():
    return 1


class Example(BaseModel):
    name: str = 'example'
    function: Callable = example_callable


instance_example = Example()

validation_schema = instance_example.model_json_schema(
    schema_generator=MyGenerateJsonSchema, mode='validation'
)
print(validation_schema)
```

LANGUAGE: json
CODE:
```
{
    'properties': {
        'name': {'default': 'example', 'title': 'Name', 'type': 'string'}
    },
    'title': 'Example',
    'type': 'object',
}
```

----------------------------------------

TITLE: Pydantic Badges in Markdown
DESCRIPTION: Demonstrates how to embed Pydantic version badges into Markdown files. These badges link to the Pydantic documentation and display the current version status.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/contributing.md#_snippet_12

LANGUAGE: md
CODE:
```
[![Pydantic v1](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v1.json)](https://pydantic.dev)

[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://pydantic.dev)
```

----------------------------------------

TITLE: Basic Pydantic Dataclass Usage
DESCRIPTION: Demonstrates defining a Pydantic dataclass with basic fields and instantiating it, showcasing automatic type coercion.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/dataclasses.md#_snippet_0

LANGUAGE: python
CODE:
```
from datetime import datetime
from typing import Optional

from pydantic.dataclasses import dataclass


@dataclass
class User:
    id: int
    name: str = 'John Doe'
    signup_ts: Optional[datetime] = None


user = User(id='42', signup_ts='2032-06-21T12:00')
print(user)
```

----------------------------------------

TITLE: Pydantic Reusable Validators with Annotated Pattern
DESCRIPTION: Shows how to create reusable validators using the `Annotated` pattern in Pydantic with `AfterValidator`. This example defines an `is_even` validator and applies it to different fields and types, including lists, promoting code reuse and cleaner type definitions.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/validators.md#_snippet_10

LANGUAGE: python
CODE:
```
from typing import Annotated

from pydantic import AfterValidator, BaseModel


def is_even(value: int) -> int:
    if value % 2 == 1:
        raise ValueError(f'{value} is not an even number')
    return value


EvenNumber = Annotated[int, AfterValidator(is_even)]


class Model1(BaseModel):
    my_number: EvenNumber


class Model2(BaseModel):
    other_number: Annotated[EvenNumber, AfterValidator(lambda v: v + 2)]


class Model3(BaseModel):
    list_of_even_numbers: list[EvenNumber]  # (1)!


# 1. As mentioned in the [annotated pattern](./fields.md#the-annotated-pattern) documentation,
#    we can also make use of validators for specific parts of the annotation (in this case,
#    validation is applied for list items, but not the whole list).
```

----------------------------------------

TITLE: Authenticate GitHub CLI
DESCRIPTION: Authenticates the GitHub CLI with your GitHub account. This is required for making API calls during the release process.

SOURCE: https://github.com/pydantic/pydantic/blob/main/release/README.md#_snippet_0

LANGUAGE: shell
CODE:
```
gh auth login
```

----------------------------------------

TITLE: Handle Pydantic JSON Validation Errors
DESCRIPTION: Illustrates how Pydantic handles invalid JSON data by raising a `ValidationError`. This example shows a JSON object with missing fields, incorrect types, and invalid formats, and how to catch and print the detailed error.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/examples/files.md#_snippet_1

LANGUAGE: python
CODE:
```
import pathlib

from pydantic import BaseModel, EmailStr, PositiveInt, ValidationError


class Person(BaseModel):
    name: str
    age: PositiveInt
    email: EmailStr


json_string = pathlib.Path('person.json').read_text()
try:
    person = Person.model_validate_json(json_string)
except ValidationError as err:
    print(err)
    """
    3 validation errors for Person
    name
    Field required [type=missing, input_value={'age': -30, 'email': 'not-an-email-address'}, input_type=dict]
        For further information visit https://errors.pydantic.dev/2.10/v/missing
    age
    Input should be greater than 0 [type=greater_than, input_value=-30, input_type=int]
        For further information visit https://errors.pydantic.dev/2.10/v/greater_than
    email
    value is not a valid email address: An email address must have an @-sign. [type=value_error, input_value='not-an-email-address', input_type=str]
    """
```

----------------------------------------

TITLE: TypeAdapter Schema Rebuilding with Deferred Build
DESCRIPTION: Shows how to defer the building of a TypeAdapter's core schema using `defer_build=True` and manually trigger it later with the `rebuild` method. This is useful for types with forward references or expensive schema builds.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/type_adapter.md#_snippet_2

LANGUAGE: python
CODE:
```
from pydantic import ConfigDict, TypeAdapter

ta = TypeAdapter('MyInt', config=ConfigDict(defer_build=True))
```

----------------------------------------

TITLE: Pydantic Serialization Alias Only
DESCRIPTION: Shows how to use `serialization_alias` in Pydantic's `Field` for an alias used exclusively during serialization. The example demonstrates validation using the field name and serialization using the specified alias.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/fields.md#_snippet_13

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field


class User(BaseModel):
    name: str = Field(serialization_alias='username')


user = User(name='johndoe')  # (1)!
print(user)
#> name='johndoe'
print(user.model_dump(by_alias=True))  # (2)!
#>
#> {'username': 'johndoe'}
```

----------------------------------------

TITLE: Run Code Formatting and Linting
DESCRIPTION: Executes automated code formatting and linting checks using tools like ruff. This command ensures code style consistency across the project.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/contributing.md#_snippet_6

LANGUAGE: bash
CODE:
```
make format
```

----------------------------------------

TITLE: Pydantic value_error Error Example
DESCRIPTION: Illustrates the 'value_error' exception, which occurs when a ValueError is raised during Pydantic field validation. Features a BaseModel with a custom validator that intentionally raises a ValueError, and shows how to capture this error.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_101

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ValidationError, field_validator


class Model(BaseModel):
    x: str

    @field_validator('x')
    @classmethod
    def repeat_b(cls, v):
        raise ValueError()


try:
    Model(x='test')
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'value_error'
```

----------------------------------------

TITLE: Custom ThirdPartyType with Pydantic
DESCRIPTION: Demonstrates how to integrate a custom third-party type with Pydantic models. Shows instance creation, attribute access, model dumping, and validation error handling for invalid inputs. Includes an example of generating the JSON schema for a model with a custom type.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/types.md#_snippet_15

LANGUAGE: python
CODE:
```
from typing import Annotated

from pydantic import BaseModel, ValidationError
from pydantic_core import core_schema

# Assume ThirdPartyType is defined elsewhere and behaves like this:
class ThirdPartyType:
    def __init__(self):
        self.x = 0

class Model(BaseModel):
    third_party_type: ThirdPartyType

# Example usage:
instance = ThirdPartyType()
assert instance.x == 0
instance.x = 10

m_instance = Model(third_party_type=instance)
assert isinstance(m_instance.third_party_type, ThirdPartyType)
assert m_instance.third_party_type.x == 10
assert m_instance.model_dump() == {'third_party_type': 10}

# Demonstrate that validation errors are raised as expected for invalid inputs
try:
    Model(third_party_type='a')
except ValidationError as e:
    print(e)
    # Expected output:
    # 2 validation errors for Model
    # third_party_type.is-instance[ThirdPartyType]
    #   Input should be an instance of ThirdPartyType [type=is_instance_of, input_value='a', input_type=str]
    # third_party_type.chain[int,function-plain[validate_from_int()]]
    #   Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='a', input_type=str]

# Example of model_json_schema
assert Model.model_json_schema() == {
    'properties': {
        'third_party_type': {'title': 'Third Party Type', 'type': 'integer'}
    },
    'required': ['third_party_type'],
    'title': 'Model',
    'type': 'object',
}

# This approach can be used for types like Pandas or Numpy types.
```

----------------------------------------

TITLE: Pydantic Dotenv Path Support
DESCRIPTION: Supports home directory relative paths (e.g., `~/.env`) for `dotenv` files used by `BaseSettings`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_204

LANGUAGE: python
CODE:
```
from pydantic import BaseSettings

class AppSettings(BaseSettings):
    database_url: str

    class Config:
        # Loads from ~/.env if it exists
        env_file = '~/.env'
        env_file_encoding = 'utf-8'

# settings = AppSettings()
# print(settings.database_url)
```

----------------------------------------

TITLE: Pydantic Forward Annotations with PEP563
DESCRIPTION: Demonstrates using `from __future__ import annotations` to enable forward references for type hints in Pydantic models. This allows referencing types that are not yet defined.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/forward_annotations.md#_snippet_0

LANGUAGE: python
CODE:
```
from __future__ import annotations

from pydantic import BaseModel

MyInt = int


class Model(BaseModel):
    a: MyInt
    # Without the future import, equivalent to:
    # a: 'MyInt'


print(Model(a='1'))
#> a=1
```

----------------------------------------

TITLE: Import Pydantic V1 Utility Function
DESCRIPTION: Demonstrates importing a utility function, 'lenient_isinstance', from the Pydantic V1 namespace. This allows access to functions removed or changed in Pydantic V2.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_5

LANGUAGE: python
CODE:
```
from pydantic.v1.utils import lenient_isinstance
```

----------------------------------------

TITLE: Python Annotation Resolution at Class Definition
DESCRIPTION: Provides a reference example for understanding how Pydantic resolves type annotations at class definition time. It sets up a scenario with a type alias and a base class using forward references, which is crucial for runtime evaluation.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/internals/resolving_annotations.md#_snippet_2

LANGUAGE: Python
CODE:
```
# module1.py:
type MyType = int

class Base:
    f1: 'MyType'
```

----------------------------------------

TITLE: decimal_type Pydantic Validation Example
DESCRIPTION: Illustrates the 'decimal_type' error, raised when the input value's type is incorrect for a Decimal field, such as providing a list. It also applies to strict fields not being instances of Decimal.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_28

LANGUAGE: python
CODE:
```
from decimal import Decimal

from pydantic import BaseModel, Field, ValidationError


class Model(BaseModel):
    x: Decimal = Field(decimal_places=3)


try:
    Model(x=[1, 2, 3])
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'decimal_type'
```

----------------------------------------

TITLE: Custom Generic Sequence Type with Pydantic
DESCRIPTION: Demonstrates creating a custom generic sequence type (`MySequence`) that integrates with Pydantic's `__get_pydantic_core_schema__` for validation and generic type handling. Includes examples of default values and validation with specific types, showing how Pydantic generates schemas for generic types.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/types.md#_snippet_24

LANGUAGE: python
CODE:
```
from typing import Any, Sequence, TypeVar

from pydantic_core import ValidationError, core_schema
from typing_extensions import get_args

from pydantic import BaseModel, GetCoreSchemaHandler

T = TypeVar('T')


class MySequence(Sequence[T]):
    def __init__(self, v: Sequence[T]):
        self.v = v

    def __getitem__(self, i):
        return self.v[i]

    def __len__(self):
        return len(self.v)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        instance_schema = core_schema.is_instance_schema(cls)

        args = get_args(source)
        if args:
            # replace the type and rely on Pydantic to generate the right schema
            # for `Sequence`
            sequence_t_schema = handler.generate_schema(Sequence[args[0]])
        else:
            sequence_t_schema = handler.generate_schema(Sequence)

        non_instance_schema = core_schema.no_info_after_validator_function(
            MySequence, sequence_t_schema
        )
        return core_schema.union_schema([instance_schema, non_instance_schema])


class M(BaseModel):
    model_config = dict(validate_default=True)

    s1: MySequence = [3]


m = M()
print(m)
#> s1=<__main__.MySequence object at 0x0123456789ab>
print(m.s1.v)
#> [3]


class M(BaseModel):
    s1: MySequence[int]


M(s1=[1])
try:
    M(s1=['a'])
except ValidationError as exc:
    print(exc)
    """
    2 validation errors for M
    s1.is-instance[MySequence]
      Input should be an instance of MySequence [type=is_instance_of, input_value=['a'], input_type=list]
    s1.function-after[MySequence(), json-or-python[json=list[int],python=chain[is-instance[Sequence],function-wrap[sequence_validator()]]]].0
      Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='a', input_type=str]
    """

```

----------------------------------------

TITLE: Configure Serverless Framework for Metadata (YAML)
DESCRIPTION: This YAML configuration snippet shows how to set `slim: false` in `serverless.yml` for the `serverless-python-requirements` plugin. This ensures that package metadata, like `dist-info` directories, is included in the deployment package, resolving issues with libraries like `email-validator` on AWS Lambda.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/integrations/aws_lambda.md#_snippet_3

LANGUAGE: yaml
CODE:
```
pythonRequirements:
    dockerizePip: non-linux
    slim: false
    fileName: requirements.txt
```

----------------------------------------

TITLE: datetime_type Pydantic Validation Example
DESCRIPTION: Demonstrates the 'datetime_type' error, raised when the input value's type is incorrect for a datetime field, such as providing None. It also applies to strict fields not being instances of datetime.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_24

LANGUAGE: python
CODE:
```
from datetime import datetime

from pydantic import BaseModel, ValidationError


class Model(BaseModel):
    x: datetime


try:
    Model(x=None)
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'datetime_type'
```

----------------------------------------

TITLE: Pydantic Field Ordering and Validation
DESCRIPTION: Explains that Pydantic preserves field order in JSON Schema, validation errors, and serialization. The example shows how fields are ordered in `model_dump()` and how validation errors report locations based on this order.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_52

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ValidationError


class Model(BaseModel):
    a: int
    b: int = 2
    c: int = 1
    d: int = 0
    e: float


print(Model.model_fields.keys())
#> dict_keys(['a', 'b', 'c', 'd', 'e'])
m = Model(e=2, a=1)
print(m.model_dump())
#> {'a': 1, 'b': 2, 'c': 1, 'd': 0, 'e': 2.0}
try:
    Model(a='x', b='x', c='x', d='x', e='x')
except ValidationError as err:
    error_locations = [e['loc'] for e in err.errors()]

print(error_locations)
#> [('a',), ('b',), ('c',), ('d',), ('e',)]
```

----------------------------------------

TITLE: Pydantic BaseModel Method Renaming (V1 to V2)
DESCRIPTION: Lists common Pydantic BaseModel methods and their corresponding names in Pydantic V2. This helps users migrate from V1 to V2 by understanding the new API.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_8

LANGUAGE: APIDOC
CODE:
```
Pydantic BaseModel Method Mapping:

| Pydantic V1 Method Name      | Pydantic V2 Method Name      |
|------------------------------|------------------------------|
| `__fields__`                 | `model_fields`               |
| `__private_attributes__`     | `__pydantic_private__`       |
| `__validators__`             | `__pydantic_validator__`     |
| `construct()`                | `model_construct()`          |
| `copy()`                     | `model_copy()`               |
| `dict()`                     | `model_dump()`               |
| `json_schema()`              | `model_json_schema()`        |
| `json()`                     | `model_dump_json()`          |
| `parse_obj()`                | `model_validate()`           |
| `update_forward_refs()`      | `model_rebuild()`            |
```

----------------------------------------

TITLE: Configuration for TypedDict using with_config
DESCRIPTION: Illustrates configuring `TypedDict` using the `with_config` decorator with `ConfigDict` to enable `str_to_lower`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/config.md#_snippet_5

LANGUAGE: python
CODE:
```
from typing_extensions import TypedDict

from pydantic import ConfigDict, with_config


@with_config(ConfigDict(str_to_lower=True))
class Model(TypedDict):
    x: str

```

----------------------------------------

TITLE: Pydantic: Customizing Settings Sources
DESCRIPTION: Provides the ability to customize how Pydantic loads settings by adding, disabling, or changing the priority order of configuration sources.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_174

LANGUAGE: APIDOC
CODE:
```
Pydantic Settings Management:

Customize settings loading behavior by modifying the `settings_customise_sources` class method or attribute within a `BaseSettings` subclass.

Functionality:
- Add new sources (e.g., custom file formats, environment variables with prefixes).
- Disable default sources (e.g., `.env` files, environment variables).
- Change the priority order of existing sources.

Example:
```python
from pydantic import BaseSettings, SettingsConfigDict
from pydantic_settings import SettingsSources, init_settings

class CustomSettings(BaseSettings):
    api_key: str
    debug_mode: bool = False

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: SettingsSources,
        env_settings: SettingsSources,
        dotenv_settings: SettingsSources,
        file_secret_settings: SettingsSources,
    ) -> SettingsSources:
        # Prioritize environment variables, then a custom config file, then .env
        return init_settings.with_priority(10) | env_settings.with_priority(20) | dotenv_settings.with_priority(30)

# Usage:
# settings = CustomSettings()
```

Related Configuration:
- `SettingsConfigDict`: Used to configure settings behavior, including source customization.
```

----------------------------------------

TITLE: Pydantic uuid_version Error Example
DESCRIPTION: Demonstrates the 'uuid_version' error raised by Pydantic when an input value's type does not match the expected UUID version. Includes a BaseModel with UUID5 and a try-except block to catch and display the error type.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_100

LANGUAGE: python
CODE:
```
from pydantic import UUID5, BaseModel, ValidationError


class Model(BaseModel):
    u: UUID5


try:
    Model(u='a6cc5730-2261-11ee-9c43-2eb5a363657c')
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'uuid_version'
```

----------------------------------------

TITLE: Pydantic Field Alias for Validation and Serialization
DESCRIPTION: Demonstrates using the `alias` parameter in Pydantic's `Field` to specify a name for both validation and serialization. The example shows creating a model instance with the alias and dumping it back to a dictionary using the alias.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/fields.md#_snippet_11

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field


class User(BaseModel):
    name: str = Field(alias='username')


user = User(username='johndoe')  # (1)!
print(user)
#> name='johndoe'
print(user.model_dump(by_alias=True))  # (2)!
#>
#> {'username': 'johndoe'}
```

----------------------------------------

TITLE: Demonstrate date_past error in Pydantic
DESCRIPTION: Illustrates the `date_past` error, raised when a value for a `PastDate` field is not in the past. The example defines a model with a `PastDate` field and attempts to validate a date that is in the future, triggering the validation error.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_17

LANGUAGE: python
CODE:
```
from datetime import date, timedelta

from pydantic import BaseModel, PastDate, ValidationError


class Model(BaseModel):
    x: PastDate


try:
    Model(x=date.today() + timedelta(1))
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'date_past'
```

----------------------------------------

TITLE: Dynamic Model with Aliases, Descriptions, and Private Attributes
DESCRIPTION: Demonstrates creating a dynamic Pydantic model using `create_model`, specifying field aliases, descriptions via `Annotated`, and private attributes.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_43

LANGUAGE: python
CODE:
```
from typing import Annotated

from pydantic import BaseModel, Field, PrivateAttr, create_model

DynamicModel = create_model(
    'DynamicModel',
    foo=(str, Field(alias='FOO')),
    bar=Annotated[str, Field(description='Bar field')],
    _private=(int, PrivateAttr(default=1)),
)
```

----------------------------------------

TITLE: Demonstrate dataclass_exact_type error in Pydantic
DESCRIPTION: Illustrates the `dataclass_exact_type` error, occurring with `strict=True` validation when the input is not an instance of the target dataclass. The example uses `TypeAdapter` with a Pydantic dataclass and shows how providing a dictionary with `strict=True` raises this specific error.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_11

LANGUAGE: python
CODE:
```
import pydantic.dataclasses
from pydantic import TypeAdapter, ValidationError


@pydantic.dataclasses.dataclass
class MyDataclass:
    x: str


adapter = TypeAdapter(MyDataclass)

print(adapter.validate_python(MyDataclass(x='test'), strict=True))
#> MyDataclass(x='test')
print(adapter.validate_python({'x': 'test'}))
#> MyDataclass(x='test')

try:
    adapter.validate_python({'x': 'test'}, strict=True)
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'dataclass_exact_type'
```

----------------------------------------

TITLE: Handle iterable_type Error in Pydantic
DESCRIPTION: This error is raised when the input value is not valid as an `Iterable` for a Pydantic field. The example demonstrates passing an integer (`123`) to a field annotated with `Iterable`, causing a `ValidationError` with the 'iterable_type' error.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_49

LANGUAGE: python
CODE:
```
from typing import Iterable

from pydantic import BaseModel, ValidationError


class Model(BaseModel):
    y: Iterable


try:
    Model(y=123)
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'iterable_type'
```

----------------------------------------

TITLE: Handle int_type Error in Pydantic
DESCRIPTION: This error is raised when the input value's type is not valid for an `int` field in Pydantic. The example demonstrates passing `None` to an `int` field, which results in a `ValidationError` with the 'int_type' error.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_45

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ValidationError


class Model(BaseModel):
    x: int


try:
    Model(x=None)
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'int_type'
```

----------------------------------------

TITLE: BaseSettings Signature
DESCRIPTION: Change in how `Any` type is handled when synthesizing the `BaseSettings.__init__` signature in the mypy plugin.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_18

LANGUAGE: python
CODE:
```
Change type of `Any` when synthesizing `BaseSettings.__init__` signature in the mypy plugin
```

----------------------------------------

TITLE: Pydantic v1 @validator Deprecation Example
DESCRIPTION: Illustrates the deprecated @validator decorator with the 'always=True' argument. It explains that this behavior is now handled by 'validate_default' in Pydantic v2's Field, and notes changes to validator signatures, removing 'field' and 'config' arguments.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_17

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, validator


class Model(BaseModel):
    x: str = 1

    @validator('x', always=True)
    @classmethod
    def validate_x(cls, v):
        return v


Model()
```

----------------------------------------

TITLE: Pydantic `create_model()` Enhancements
DESCRIPTION: The `create_model()` function now allows specifying configuration and base models together. This provides more flexibility when dynamically creating Pydantic models, enabling simultaneous definition of model settings and inheritance.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_1

LANGUAGE: python
CODE:
```
from pydantic import create_model, ConfigDict

# Example: Creating a model with config and bases specified together
MyDynamicModel = create_model(
    'MyDynamicModel',
    __config__=ConfigDict(extra='forbid'),
    field1=(str, ...),
    base_model_field=(int, 10)
)

# This is equivalent to:
# class MyDynamicModel(BaseModel):
#     field1: str
#     base_model_field: int = 10
#     model_config = ConfigDict(extra='forbid')

```

----------------------------------------

TITLE: Global Configuration Inheritance (Allow Extra)
DESCRIPTION: Demonstrates setting global configuration `extra='allow'` in a parent `BaseModel` and inheriting it in a child model.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/config.md#_snippet_6

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ConfigDict


class Parent(BaseModel):
    model_config = ConfigDict(extra='allow')


class Model(Parent):
    x: str


m = Model(x='foo', y='bar')
print(m.model_dump())
#> {'x': 'foo', 'y': 'bar'}

```

----------------------------------------

TITLE: Pydantic Validation Alias Only
DESCRIPTION: Illustrates using `validation_alias` in Pydantic's `Field` to specify an alias used only during model validation. The example shows instance creation with the validation alias and serialization using the original field name.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/fields.md#_snippet_12

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field


class User(BaseModel):
    name: str = Field(validation_alias='username')


user = User(username='johndoe')  # (1)!
print(user)
#> name='johndoe'
print(user.model_dump(by_alias=True))  # (2)!
#>
#> {'name': 'johndoe'}
```

----------------------------------------

TITLE: Pydantic Model with Dict for Nested Model
DESCRIPTION: Demonstrates a Pydantic `Quest` model that expects a `Knight` model instance for its `knight` field. The example shows that passing a literal dictionary that matches the `Knight` structure is still valid and automatically converted by Pydantic, but may trigger strict type errors in linters.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/integrations/visual_studio_code.md#_snippet_2

LANGUAGE: python
CODE:
```
from pydantic import BaseModel


class Knight(BaseModel):
    title: str
    age: int
    color: str = 'blue'


class Quest(BaseModel):
    title: str
    knight: Knight


quest = Quest(
    title='To seek the Holy Grail', knight={'title': 'Sir Lancelot', 'age': 23}
)
```

----------------------------------------

TITLE: Named Type Alias with Type Variables (Python 3.12+)
DESCRIPTION: Shows the usage of generic named type aliases with the `type` statement syntax, incorporating `TypeVar`. This example defines `ShortList` as a list of type `T` with a maximum length constraint, demonstrating parameterization.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/types.md#_snippet_9

LANGUAGE: Python
CODE:
```
from typing import Annotated, TypeVar

from annotated_types import Len

type ShortList[T] = Annotated[list[T], Len(max_length=4)]
```

----------------------------------------

TITLE: Avoid Naming Collisions with Types
DESCRIPTION: Warns about potential issues when a field name clashes with its type annotation in Python. This example shows how such a collision can lead to validation errors due to how Python evaluates annotated assignment statements.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_7

LANGUAGE: python
CODE:
```
from typing import Optional

from pydantic import BaseModel


class Boo(BaseModel):
    int: Optional[int] = None


m = Boo(int=123)  # Will fail to validate.
```

----------------------------------------

TITLE: Pydantic Model Configuration with model_config
DESCRIPTION: Demonstrates configuring a Pydantic BaseModel using the `model_config` class attribute with `ConfigDict` to set `str_max_length`. Includes error handling for validation.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/config.md#_snippet_0

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ConfigDict, ValidationError


class Model(BaseModel):
    model_config = ConfigDict(str_max_length=5)  # (1)!

    v: str


try:
    m = Model(v='abcdef')
except ValidationError as e:
    print(e)
    """
    1 validation error for Model
    v
      String should have at most 5 characters [type=string_too_long, input_value='abcdef', input_type=str]
    """

```

----------------------------------------

TITLE: Pydantic: Use Context in Validators with ValidationInfo.context
DESCRIPTION: Shows how to pass and access custom context data in Pydantic validators using `ValidationInfo.context`. This allows for dynamic behavior based on external data, requiring `pydantic` and `ValidationInfo`. The example demonstrates removing stopwords from text based on provided context.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/validators.md#_snippet_17

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ValidationInfo, field_validator


class Model(BaseModel):
    text: str

    @field_validator('text', mode='after')
    @classmethod
    def remove_stopwords(cls, v: str, info: ValidationInfo) -> str:
        if isinstance(info.context, dict):
            stopwords = info.context.get('stopwords', set())
            v = ' '.join(w for w in v.split() if w.lower() not in stopwords)
        return v


data = {'text': 'This is an example document'}
print(Model.model_validate(data))  # no context
#> text='This is an example document'
print(Model.model_validate(data, context={'stopwords': ['this', 'is', 'an']}))
#> text='example document'
```

----------------------------------------

TITLE: Pydantic Forward Reference Error Example
DESCRIPTION: Demonstrates a scenario where a Pydantic model uses a forward reference ('A | Forward') that might lead to a PydanticUndefinedAnnotation error if not properly handled during model rebuilding. It shows how to define a model within a function and the subsequent rebuild process.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/internals/resolving_annotations.md#_snippet_4

LANGUAGE: python
CODE:
```
def func():
    A = int

    class Model(BaseModel):
        f: 'A | Forward'

    return Model


Model = func()

Model.model_rebuild(_types_namespace={'Forward': str})
# pydantic.errors.PydanticUndefinedAnnotation: name 'A' is not defined
```

----------------------------------------

TITLE: TypeAdapter with Generic Collections
DESCRIPTION: Demonstrates how Pydantic V2's TypeAdapter handles generic collections, showing that input types are not preserved by default, and a plain dict is returned instead of a custom dict subclass.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_20

LANGUAGE: python
CODE:
```
from typing import Mapping

from pydantic import TypeAdapter


class MyDict(dict):
    pass


ta = TypeAdapter(Mapping[str, int])
v = ta.validate_python(MyDict())
print(type(v))
#> <class 'dict'>
```

----------------------------------------

TITLE: Customize JSON Schema $ref Paths with Pydantic
DESCRIPTION: This Python example demonstrates how to use the 'ref_template' argument with Pydantic's 'json_schema' method (via TypeAdapter) to alter the '$ref' paths in the generated JSON schema. This is useful for aligning with specific schema conventions, such as those found in OpenAPI's 'components/schemas' section.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/json_schema.md#_snippet_33

LANGUAGE: python
CODE:
```
import json

from pydantic import BaseModel
from pydantic.type_adapter import TypeAdapter


class Foo(BaseModel):
    a: int


class Model(BaseModel):
    a: Foo


adapter = TypeAdapter(Model)

print(
    json.dumps(
        adapter.json_schema(ref_template='#/components/schemas/{model}'),
        indent=2,
    )
)
"""
{
  "$defs": {
    "Foo": {
      "properties": {
        "a": {
          "title": "A",
          "type": "integer"
        }
      },
      "required": [
        "a"
      ],
      "title": "Foo",
      "type": "object"
    }
  },
  "properties": {
    "a": {
      "$ref": "#/components/schemas/Foo"
    }
  },
  "required": [
    "a"
  ],
  "title": "Model",
  "type": "object"
}
"""
```

----------------------------------------

TITLE: Pydantic Model Configuration with Class Arguments
DESCRIPTION: Shows how to configure a Pydantic BaseModel using class arguments, specifically `frozen=True`, which is recognized by static type checkers.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/config.md#_snippet_1

LANGUAGE: python
CODE:
```
from pydantic import BaseModel


class Model(BaseModel, frozen=True):
    a: str  # (1)!

```

----------------------------------------

TITLE: Verify Python Version Suffix (Python)
DESCRIPTION: This Python code uses `sysconfig` to retrieve the expected shared library suffix for the current Python environment. It's used to confirm compatibility with the compiled `pydantic_core` library, ensuring the correct native code is installed for the target platform.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/integrations/aws_lambda.md#_snippet_2

LANGUAGE: python
CODE:
```
import sysconfig
print(sysconfig.get_config_var("EXT_SUFFIX"))
#> '.cpython-312-x86_64-linux-gnu.so'
```

----------------------------------------

TITLE: Handle invalid_key Error in Pydantic
DESCRIPTION: This error is raised when attempting to validate a `dict` that contains a key which is not an instance of `str`. The example shows passing a bytes key (`b'y'`) to a Pydantic model, triggering the 'invalid_key' validation error.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_46

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ConfigDict, ValidationError


class Model(BaseModel):
    x: int

    model_config = ConfigDict(extra='allow')


try:
    Model.model_validate({'x': 1, b'y': 2})
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'invalid_key'
```

----------------------------------------

TITLE: Modify JSON Schema with __get_pydantic_json_schema__
DESCRIPTION: Shows how to customize the JSON schema generated by Pydantic by implementing the `__get_pydantic_json_schema__` class method. This method allows adding metadata like `examples` or changing the `title` without affecting the core validation schema.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/json_schema.md#_snippet_18

LANGUAGE: python
CODE:
```
import json
from typing import Any

from pydantic_core import core_schema as cs

from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler, TypeAdapter
from pydantic.json_schema import JsonSchemaValue


class Person:
    name: str
    age: int

    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> cs.CoreSchema:
        return cs.typed_dict_schema(
            {
                'name': cs.typed_dict_field(cs.str_schema()),
                'age': cs.typed_dict_field(cs.int_schema()),
            },
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        core_schema: cs.CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        json_schema = handler(core_schema)
        json_schema = handler.resolve_ref_schema(json_schema)
        json_schema['examples'] = [
            {
                'name': 'John Doe',
                'age': 25,
            }
        ]
        json_schema['title'] = 'Person'
        return json_schema


print(json.dumps(TypeAdapter(Person).json_schema(), indent=2))

```

----------------------------------------

TITLE: Enforce Typing Relationships with Nested Generics
DESCRIPTION: Demonstrates using the same type variable across nested generic models to enforce consistent typing. Includes an example of how mismatched types lead to `ValidationError` and how Pydantic handles revalidation for intuitive results.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_32

LANGUAGE: python
CODE:
```
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar('T')


class InnerT(BaseModel, Generic[T]):
    inner: T


class OuterT(BaseModel, Generic[T]):
    outer: T
    nested: InnerT[T]


nested = InnerT[int](inner=1)
print(OuterT[int](outer=1, nested=nested))
#> outer=1 nested=InnerT[int](inner=1)
try:
    print(OuterT[int](outer='a', nested=InnerT(inner='a')))  # (1)!
except ValidationError as e:
    print(e)
    """
    2 validation errors for OuterT[int]
    outer
      Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='a', input_type=str]
    nested.inner
      Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='a', input_type=str]
    """

```

----------------------------------------

TITLE: Pydantic Dataclass with Fields and Metadata
DESCRIPTION: Shows how to use `dataclasses.field` and Pydantic's `Field` for default values, factories, and metadata like titles and constraints.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/dataclasses.md#_snippet_1

LANGUAGE: python
CODE:
```
import dataclasses
from typing import Optional

from pydantic import Field
from pydantic.dataclasses import dataclass


@dataclass
class User:
    id: int
    name: str = 'John Doe'
    friends: list[int] = dataclasses.field(default_factory=lambda: [0])
    age: Optional[int] = dataclasses.field(
        default=None,
        metadata={'title': 'The age of the user', 'description': 'do not lie!'},
    )
    height: Optional[int] = Field(
        default=None, title='The height in cm', ge=50, le=300
    )


user = User(id='42', height='250')
print(user)
```

----------------------------------------

TITLE: Improving Pydantic Union Error Clarity with Tags
DESCRIPTION: This example demonstrates how `Tag` can be used with `TypeAdapter` to make validation error messages for union types more informative. It compares error outputs when `Tag` is not used versus when it is, highlighting how tags provide clearer context for which union member failed validation.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/unions.md#_snippet_12

LANGUAGE: python
CODE:
```
from typing import Annotated, Union

from pydantic import AfterValidator, Tag, TypeAdapter, ValidationError

DoubledList = Annotated[list[int], AfterValidator(lambda x: x * 2)]
StringsMap = dict[str, str]


# Not using any `Tag`s for each union case, the errors are not so nice to look at
adapter = TypeAdapter(Union[DoubledList, StringsMap])

try:
    adapter.validate_python(['a'])
except ValidationError as exc_info:
    print(exc_info)

tag_adapter = TypeAdapter(
    Union[
        Annotated[DoubledList, Tag('DoubledList')],
        Annotated[StringsMap, Tag('StringsMap')],
    ]
)

try:
    tag_adapter.validate_python(['a'])
except ValidationError as exc_info:
    print(exc_info)
```

----------------------------------------

TITLE: Define Pydantic User Model
DESCRIPTION: Demonstrates how to define a Pydantic BaseModel with required and optional fields, default values, and model configuration. It shows the basic structure for creating data models that leverage Python's type hints for validation.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_2

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ConfigDict


class User(BaseModel):
    id: int
    name: str = 'Jane Doe'

    model_config = ConfigDict(str_max_length=10)  # (1)!
```

----------------------------------------

TITLE: Demonstrate datetime_future error in Pydantic
DESCRIPTION: Illustrates the `datetime_future` error, raised when a value provided for a `FutureDatetime` field is not in the future. The example defines a model with a `FutureDatetime` field and attempts to validate a past `datetime` object, triggering the validation error.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_20

LANGUAGE: python
CODE:
```
from datetime import datetime

from pydantic import BaseModel, FutureDatetime, ValidationError


class Model(BaseModel):
    x: FutureDatetime


try:
    Model(x=datetime(2000, 1, 1))
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'datetime_future'
```

----------------------------------------

TITLE: TypeAdapter: Explicit Generic Parameter Specification
DESCRIPTION: Illustrates the need for explicit generic parameter specification with `TypeAdapter` in certain scenarios to ensure proper typing, especially when dealing with complex union types.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_28

LANGUAGE: python
CODE:
```
from pydantic import TypeAdapter

adapter = TypeAdapter[str | int](str | int)
...
```

----------------------------------------

TITLE: Pydantic Generic Model Revalidation Behavior
DESCRIPTION: Explains Pydantic's revalidation strategy for nested generic models, particularly when validating data against a generic type like `GenericModel[Any]`. Shows an example where revalidation ensures intuitive results for compatible types.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_33

LANGUAGE: python
CODE:
```
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar('T')


class GenericModel(BaseModel, Generic[T]):
    a: T


class Model(BaseModel):
    inner: GenericModel[Any]


print(repr(Model.model_validate(Model(inner=GenericModel[int](a=1)))))
#> Model(inner=GenericModel[Any](a=1))
```

----------------------------------------

TITLE: Stdlib Type Configuration Propagation with Pydantic
DESCRIPTION: Shows that configuration is propagated for standard library types (dataclasses, typed dictionaries) unless they have their own configuration set, when used with Pydantic.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/config.md#_snippet_9

LANGUAGE: python
CODE:
```
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, with_config


@dataclass
class UserWithoutConfig:
    name: str


@dataclass
@with_config(str_to_lower=False)
class UserWithConfig:
    name: str


class Parent(BaseModel):
    user_1: UserWithoutConfig
    user_2: UserWithConfig

    model_config = ConfigDict(str_to_lower=True)


print(Parent(user_1={'name': 'JOHN'}, user_2={'name': 'JOHN'}))
#> user_1=UserWithoutConfig(name='john') user_2=UserWithConfig(name='JOHN')
```

----------------------------------------

TITLE: Mark Pydantic `computed_field` as Deprecated
DESCRIPTION: This example demonstrates how to mark a `computed_field` as deprecated using the `@deprecated` decorator from `typing_extensions`. This allows developers to signal that a computed property is no longer recommended for use, providing clear guidance for API evolution.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/fields.md#_snippet_41

LANGUAGE: python
CODE:
```
from typing_extensions import deprecated

from pydantic import BaseModel, computed_field


class Box(BaseModel):
    width: float
    height: float
    depth: float

    @computed_field
    @property
    @deprecated("'volume' is deprecated")
    def volume(self) -> float:
        return self.width * self.height * self.depth
```

----------------------------------------

TITLE: Handle json_invalid Error in Pydantic
DESCRIPTION: This error is raised when the input value provided to a `Json` field in Pydantic is not a valid JSON string. The example attempts to parse a plain string 'test' as JSON, triggering the 'json_invalid' validation error.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_51

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Json, ValidationError


class Model(BaseModel):
    x: Json


try:
    Model(x='test')
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'json_invalid'
```

----------------------------------------

TITLE: Pydantic Apply Validator to Multiple Fields with Decorator
DESCRIPTION: Demonstrates using the `@field_validator` decorator to apply a single validation function to multiple fields within a Pydantic model. This example uses a `capitalize` validator applied to `f1` and `f2` using `mode='before'`, showcasing efficient application of common validation logic.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/validators.md#_snippet_11

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, field_validator


class Model(BaseModel):
    f1: str
    f2: str

    @field_validator('f1', 'f2', mode='before')
    @classmethod
    def capitalize(cls, value: str) -> str:
        return value.capitalize()
```

----------------------------------------

TITLE: Handle is_subclass_of Error in Pydantic
DESCRIPTION: This error is raised when the input value is not a subclass of the expected type for a Pydantic field. The example shows passing a string to a field annotated with `type[Nested]`, which expects a class, triggering the 'is_subclass_of' error.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_48

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ValidationError


class Nested:
    x: str


class Model(BaseModel):
    y: type[Nested]


try:
    Model(y='test')
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'is_subclass_of'
```

----------------------------------------

TITLE: Strict Mode with Annotated[..., Strict()]
DESCRIPTION: Demonstrates using `Annotated[..., Strict()]` to enforce strict validation on individual fields within a Pydantic model. This example shows how a string 'True' fails validation for a strictly typed boolean field.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/strict_mode.md#_snippet_11

LANGUAGE: python
CODE:
```
from typing import Annotated

from pydantic import BaseModel, Strict, ValidationError


class User(BaseModel):
    name: str
    age: int
    is_active: Annotated[bool, Strict()]


User(name='David', age=33, is_active=True)
try:
    User(name='David', age=33, is_active='True')
except ValidationError as exc:
    print(exc)
    
```

----------------------------------------

TITLE: Customizing Core Schema with Annotated
DESCRIPTION: Illustrates how to customize Pydantic's core schema generation for types using Annotated and custom classes like MyStrict and MyGt, which implement __get_pydantic_core_schema__.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/internals/architecture.md#_snippet_6

LANGUAGE: python
CODE:
```
from typing import Annotated, Any

from pydantic_core import CoreSchema

from pydantic import GetCoreSchemaHandler, TypeAdapter


class MyStrict:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source: Any,
        handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        schema = handler(source)  # (1)!
        schema['strict'] = True
        return schema


class MyGt:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source: Any,
        handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        schema = handler(source)  # (2)!
        schema['gt'] = 1
        return schema


ta = TypeAdapter(Annotated[int, MyStrict(), MyGt()])
```

----------------------------------------

TITLE: Pydantic v1 Deprecation Aliases Removal
DESCRIPTION: This snippet details the removal of deprecated aliases and features from Pydantic v1, guiding users on migration paths. It lists specific attributes and methods that have been replaced or removed, such as `Schema`, `Config.case_insensitive`, `model.fields`, and `model.__values__`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_141

LANGUAGE: APIDOC
CODE:
```
Pydantic v1 Deprecation Aliases Removal:

- Removed `Schema` (replaced by `Field`)
- Removed `Config.case_insensitive` (replaced by `Config.case_sensitive`, default `False`)
- Removed `Config.allow_population_by_alias` (replaced by `Config.allow_population_by_field_name`)
- Removed `model.fields` (replaced by `model.__fields__`)
- Removed `model.to_string()` (replaced by `str(model)`)
- Removed `model.__values__` (replaced by `model.__dict__`)
- Removed notes on migrating to v1 in docs.
```

----------------------------------------

TITLE: Pydantic Validation Error Structure Example
DESCRIPTION: This snippet demonstrates the structure of a Pydantic validation error, showing common fields like 'type', 'loc', 'msg', 'input', 'ctx', and 'url'. These structures are used to convey detailed information about why data validation failed.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/errors.md#_snippet_2

LANGUAGE: Python
CODE:
```
[
    {
        'type': 'missing',
        'loc': ('body',),
        'msg': 'Field required',
        'input': {},
        'url': 'https://errors.pydantic.dev/2/v/missing',
    },
    {
        'type': 'greater_than',
        'loc': ('gt_int',),
        'msg': 'Input should be greater than 42',
        'input': 21,
        'ctx': {'gt': 42},
        'url': 'https://errors.pydantic.dev/2/v/greater_than',
    },
    {
        'type': 'int_parsing',
        'loc': ('list_of_ints', 2),
        'msg': 'Input should be a valid integer, unable to parse string as an integer',
        'input': 'bad',
        'url': 'https://errors.pydantic.dev/2/v/int_parsing',
    },
    {
        'type': 'value_error',
        'loc': ('a_float',),
        'msg': 'Value error, Invalid float value',
        'input': 3.0,
        'ctx': {'error': ValueError('Invalid float value')},
        'url': 'https://errors.pydantic.dev/2/v/value_error',
    },
    {
        'type': 'float_parsing',
        'loc': ('recursive_model', 'lng'),
        'msg': 'Input should be a valid number, unable to parse string as a number',
        'input': 'New York',
        'url': 'https://errors.pydantic.dev/2/v/float_parsing',
    },
]
```

----------------------------------------

TITLE: Pydantic Model Configuration via Class Kwargs
DESCRIPTION: Introduces the ability to configure Pydantic models directly through class keyword arguments. This offers a more concise and integrated way to set model-specific configurations.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_145

LANGUAGE: APIDOC
CODE:
```
Pydantic Model Configuration via Class Kwargs:

Models can now be configured using class keyword arguments.
```

----------------------------------------

TITLE: Basic Include in Pydantic
DESCRIPTION: Demonstrates how to use the `include` parameter to specify which fields to serialize. It shows including specific fields and nested fields.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/serialization.md#_snippet_21

LANGUAGE: python
CODE:
```
from pydantic import BaseModel

class User(BaseModel):
    id: str
    name: str
    user: dict

t = User(id='1234567890', name='John Doe', user={'id': 42})
print(t.model_dump(include={'id': True, 'user': {'id'}}))
#> {'id': '1234567890', 'user': {'id': 42}}
```

----------------------------------------

TITLE: Pydantic Model Configuration Propagation
DESCRIPTION: Demonstrates that configuration is not propagated across Pydantic models, with each model having its own configuration boundary.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/config.md#_snippet_8

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ConfigDict


class User(BaseModel):
    name: str


class Parent(BaseModel):
    user: User

    model_config = ConfigDict(str_to_lower=True)


print(Parent(user={'name': 'JOHN'}))
#> user=User(name='JOHN')
```

----------------------------------------

TITLE: Define Pydantic Model with `typing.Literal` for Enum-like Fields
DESCRIPTION: Demonstrates how to use `typing.Literal` in Pydantic models to restrict a field's value to a predefined set of string literals. Includes an example of successful instantiation and a `ValidationError` for an invalid input, showing how Pydantic enforces these literal constraints.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/standard_library_types.md#_snippet_28

LANGUAGE: python
CODE:
```
from typing import Literal

from pydantic import BaseModel, ValidationError


class Pie(BaseModel):
    flavor: Literal['apple', 'pumpkin']


Pie(flavor='apple')
Pie(flavor='pumpkin')
try:
    Pie(flavor='cherry')
except ValidationError as e:
    print(str(e))
    """
    1 validation error for Pie
    flavor
      Input should be 'apple' or 'pumpkin' [type=literal_error, input_value='cherry', input_type=str]
    """
```

----------------------------------------

TITLE: Sphinx Intersphinx Configuration for Pydantic
DESCRIPTION: Configure Sphinx to enable cross-referencing with Pydantic's API documentation. This involves adding Pydantic's object inventory URL to the `intersphinx_mapping` in your Sphinx configuration file (`conf.py`).

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/integrations/documentation.md#_snippet_0

LANGUAGE: Python
CODE:
```
intersphinx_mapping = {
    'pydantic': ('https://docs.pydantic.dev/latest', None)
}
```

----------------------------------------

TITLE: Handle iteration_error in Pydantic
DESCRIPTION: This error is raised when an error occurs during the iteration process for a Pydantic field, such as when validating a list from a generator that raises an exception. The example shows a generator yielding values and then raising a `RuntimeError`, which Pydantic catches and reports as 'iteration_error'.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_50

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ValidationError


def gen():
    yield 1
    raise RuntimeError('error')


class Model(BaseModel):
    x: list[int]


try:
    Model(x=gen())
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'iteration_error'
```

----------------------------------------

TITLE: Handle list_type Error in Pydantic
DESCRIPTION: This error is raised when the input value's type is not valid for a `list` field in Pydantic. The example demonstrates passing an integer (`1`) to a field annotated as `list[int]`, which causes a `ValidationError` with the 'list_type' error.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_55

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ValidationError


class Model(BaseModel):
    x: list[int]


try:
    Model(x=1)
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'list_type'
```

----------------------------------------

TITLE: Pydantic Mypy Plugin Compatibility with disallow_any_explicit
DESCRIPTION: A note on compatibility issues with mypy's `disallow_any_explicit` option. It explains that synthesized `__init__` methods might contain `Any` annotations, causing errors, and provides a solution.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/integrations/mypy.md#_snippet_7

LANGUAGE: APIDOC
CODE:
```
Compatibility Note:
  Topic: Interaction with mypy's `disallow_any_explicit`
  Problem: Synthesized `__init__` methods may contain `Any` annotations, causing errors if `disallow_any_explicit` is enabled.
  Solution: Enable both `init_forbid_extra` and `init_typed` to circumvent this issue.
```

----------------------------------------

TITLE: Handle is_instance_of Error in Pydantic
DESCRIPTION: This error occurs when an input value is not an instance of the expected type for a Pydantic field. The example demonstrates assigning a string to a field that expects a custom object type (`Nested`), resulting in an 'is_instance_of' validation error.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_47

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ConfigDict, ValidationError


class Nested:
    x: str


class Model(BaseModel):
    y: Nested

    model_config = ConfigDict(arbitrary_types_allowed=True)


try:
    Model(y='test')
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'is_instance_of'
```

----------------------------------------

TITLE: Demonstrate date_from_datetime_parsing error in Pydantic
DESCRIPTION: This example demonstrates the `date_from_datetime_parsing` error, raised when a string input for a `date` field cannot be parsed into a valid date. The code attempts to validate a Pydantic model with a `date` field using a malformed string, resulting in this parsing error.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_14

LANGUAGE: python
CODE:
```
from datetime import date

from pydantic import BaseModel, ValidationError


class Model(BaseModel):
    x: date


try:
    Model(x='XX1494012000')
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'date_from_datetime_parsing'
```

----------------------------------------

TITLE: Pydantic Enum Validation with Python Enums
DESCRIPTION: Illustrates Pydantic's integration with Python's standard `enum.Enum` and `IntEnum` classes for defining choices and validating model fields. Shows how Pydantic checks for valid Enum instances and members, including examples of successful instantiation and error handling for invalid inputs.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/standard_library_types.md#_snippet_7

LANGUAGE: python
CODE:
```
from enum import Enum, IntEnum

from pydantic import BaseModel, ValidationError


class FruitEnum(str, Enum):
    pear = 'pear'
    banana = 'banana'


class ToolEnum(IntEnum):
    spanner = 1
    wrench = 2


class CookingModel(BaseModel):
    fruit: FruitEnum = FruitEnum.pear
    tool: ToolEnum = ToolEnum.spanner


print(CookingModel())
print(CookingModel(tool=2, fruit='banana'))
try:
    CookingModel(fruit='other')
except ValidationError as e:
    print(e)
```

----------------------------------------

TITLE: Pydantic `con*` Type Functions Documentation
DESCRIPTION: Adds documentation for Pydantic's `con*` type functions, such as `conint`, `conlist`, `conset`, etc. This helps users understand and utilize these constrained types effectively.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_166

LANGUAGE: APIDOC
CODE:
```
Pydantic `con*` Type Functions Documentation:

Adds documentation for `con*` type functions.
```

----------------------------------------

TITLE: Model Validation and Serialization with pydantic-core
DESCRIPTION: Shows how Pydantic uses pydantic-core's SchemaValidator for validation and SchemaSerializer for serialization, demonstrating instance-level operations.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/internals/architecture.md#_snippet_7

LANGUAGE: python
CODE:
```
from pydantic import BaseModel


class Model(BaseModel):
    foo: int


model = Model.model_validate({'foo': 1})  # (1)!
dumped = model.model_dump()  # (2)!
```

----------------------------------------

TITLE: Pydantic: Self Type in BaseModel Method
DESCRIPTION: Shows an example where `Self` is used within a method of a `BaseModel` subclass, which is also disallowed by Pydantic's current validation rules for `Self`. This specific usage, even if type-checker valid, will raise an error.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/usage_errors.md#_snippet_55

LANGUAGE: python
CODE:
```
from typing_extensions import Self

from pydantic import BaseModel, PydanticUserError, validate_call

try:

    class A(BaseModel):
        @validate_call
        def func(self, arg: Self):
            pass

except PydanticUserError as exc_info:
    assert exc_info.code == 'invalid-self-type'
```

----------------------------------------

TITLE: Pydantic Utility Functions
DESCRIPTION: A collection of utility functions for Pydantic, including class attribute handling, value comparisons, identifier validation, and path type management.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_37

LANGUAGE: APIDOC
CODE:
```
pydantic.utils:
  ClassAttribute: A descriptor for class attributes.
  DUNDER_ATTRIBUTES: A set of dunder attributes.
  PyObjectStr: Alias for typing.Union[str, PyObject].
  ValueItems: Alias for typing.Tuple[str, Any].
  almost_equal_floats(a: float, b: float, *, tolerance: float = 1e-9) -> bool
    Check if two floats are almost equal.
  get_discriminator_alias_and_values(model: type) -> typing.Tuple[str, typing.List[typing.Any]]
    Get the discriminator alias and values for a model.
  get_model(obj: typing.Any) -> typing.Optional[type]
    Get the Pydantic model from an object.
  get_unique_discriminator_alias(model: type) -> str
    Get the unique discriminator alias for a model.
  in_ipython() -> bool
    Check if the code is running in an IPython environment.
  is_valid_identifier(name: str) -> bool
    Check if a string is a valid Python identifier.
  path_type() -> type
    Get the path type (pathlib.Path or str).
  validate_field_name(name: str) -> str
    Validate and return a field name.
```

----------------------------------------

TITLE: Generate Pydantic Models from JSON Schema
DESCRIPTION: Demonstrates the command-line usage of datamodel-code-generator to convert a JSON Schema file into Python Pydantic models. It specifies the input file, its type, and the desired output file.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/integrations/datamodel_code_generator.md#_snippet_1

LANGUAGE: bash
CODE:
```
datamodel-codegen  --input person.json --input-file-type jsonschema --output model.py
```

----------------------------------------

TITLE: Pydantic Field Serializer API Reference
DESCRIPTION: API documentation for Pydantic's field serializer components, including PlainSerializer, WrapSerializer, and the field_serializer decorator.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/serialization.md#_snippet_8

LANGUAGE: APIDOC
CODE:
```
pydantic.functional_serializers.PlainSerializer
  - Serializer that is called unconditionally.
  - The serialization logic for types supported by Pydantic will not be called.

pydantic.functional_serializers.WrapSerializer
  - Serializer that provides more flexibility by allowing custom logic before or after Pydantic's default serialization.
  - Requires a mandatory 'handler' parameter.

pydantic.functional_serializers.field_serializer
  - Decorator to apply custom serialization logic to fields.
  - Can be used with mode='plain' or mode='wrap'.
  - Applied on instance or static methods.
  - Signature: @field_serializer(field_name, *, mode='plain', return_type=None)
    - field_name: The name of the field to serialize.
    - mode: 'plain' or 'wrap'. Defaults to 'plain'.
    - return_type: Optional return type annotation for the serialized value.
```

----------------------------------------

TITLE: Support Multiple Dotenv Files
DESCRIPTION: Enhances Pydantic's settings management to support loading configuration from multiple `.env` files, allowing for layered configuration.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_107

LANGUAGE: python
CODE:
```
from pydantic import BaseSettings, SettingsConfigDict

class AppSettingsMultiEnv(BaseSettings):
    setting1: str
    setting2: int

    model_config = SettingsConfigDict(
        env_file=('.env', '.env.prod'), # Loads from .env first, then .env.prod
        env_file_encoding='utf-8'
    )

# Example:
# .env file:
# setting1=value_from_env1
# .env.prod file:
# setting2=123
# The model will load both settings.
```

----------------------------------------

TITLE: Handle literal_error in Pydantic
DESCRIPTION: This error is raised when an input value is not one of the expected literal values defined for a field using `typing.Literal`. The example shows a field `x` restricted to literals 'a' or 'b', and passing 'c' triggers the 'literal_error' validation error.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_56

LANGUAGE: python
CODE:
```
from typing import Literal

from pydantic import BaseModel, ValidationError


class Model(BaseModel):
    x: Literal['a', 'b']


Model(x='a')  # OK

try:
    Model(x='c')
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'literal_error'
```

----------------------------------------

TITLE: Handle JSON Instantiation Type Mismatch Errors
DESCRIPTION: Shows how Pydantic catches `ValidationError` when JSON data is invalid for models with custom generic types. The example uses `model_validate_json` with mismatched types in the JSON payload, demonstrating error reporting for missing fields.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/types.md#_snippet_23

LANGUAGE: python
CODE:
```
try:
    Model.model_validate_json(
        '{"car_owner":{"name":"John","item":{"rooms":3}},"home_owner":{"name":"James","item":{"color":"black"}}}'
    )
except ValidationError as e:
    print(e)

```

----------------------------------------

TITLE: Handle json_type Error in Pydantic
DESCRIPTION: This error is raised when the input value's type cannot be parsed as JSON for a `Json` field in Pydantic. The example shows passing `None` to a `Json` field, which is not a valid JSON type, resulting in a 'json_type' validation error.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_52

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Json, ValidationError


class Model(BaseModel):
    x: Json


try:
    Model(x=None)
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'json_type'
```

----------------------------------------

TITLE: Parametrize Generic Class with String
DESCRIPTION: Demonstrates parametrizing a generic class with specific types like `str` and `int`. Shows how to instantiate and represent the resulting object, highlighting type-specific initialization.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_29

LANGUAGE: python
CODE:
```
print(ChildClass[str, int](x='1', y='y', z='3'))
#> x=1 y='y' z=3
```

----------------------------------------

TITLE: Customize JSON Schema $ref with ref_template in Python
DESCRIPTION: Demonstrates customizing JSON schema $ref format using Pydantic's `ref_template` argument. This allows specifying custom paths for schema references, beneficial for OpenAPI integrations. The example shows a Pydantic model with nested structures and its generated schema with custom $ref paths.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/json_schema.md#_snippet_26

LANGUAGE: python
CODE:
```
import json

from pydantic import BaseModel
from pydantic.type_adapter import TypeAdapter


class Foo(BaseModel):
    a: int


class Model(BaseModel):
    a: Foo


adapter = TypeAdapter(Model)

print(
    json.dumps(
        adapter.json_schema(ref_template='#/components/schemas/{model}'),
        indent=2,
    )
)
```

LANGUAGE: json
CODE:
```
{
  "$defs": {
    "Foo": {
      "properties": {
        "a": {
          "title": "A",
          "type": "integer"
        }
      },
      "required": [
        "a"
      ],
      "title": "Foo",
      "type": "object"
    }
  },
  "properties": {
    "a": {
      "$ref": "#/components/schemas/Foo"
    }
  },
  "required": [
    "a"
  ],
  "title": "Model",
  "type": "object"
}
```

----------------------------------------

TITLE: TypeAdapter Configuration
DESCRIPTION: Demonstrates configuring a `TypeAdapter` with `ConfigDict` to enable `coerce_numbers_to_str` for type coercion.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/config.md#_snippet_3

LANGUAGE: python
CODE:
```
from pydantic import ConfigDict, TypeAdapter

ta = TypeAdapter(list[str], config=ConfigDict(coerce_numbers_to_str=True))

print(ta.validate_python([1, 2]))
#> ['1', '2']

```

----------------------------------------

TITLE: TypeAdapter defer_build Support
DESCRIPTION: Introduces experimental support for the 'defer_build' argument within Pydantic's TypeAdapter, allowing for deferred model building.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_36

LANGUAGE: APIDOC
CODE:
```
TypeAdapter:
  defer_build(value: Any, *, include: Include | None = None, exclude: Exclude | None = None, exclude_unset: bool = False, exclude_defaults: bool = False, exclude_none: bool = False, ...) -> Any
    Allows deferring the model building process for a TypeAdapter.
    
    Parameters:
      value: The input data to be processed.
      include: Fields to include in the output.
      exclude: Fields to exclude from the output.
      exclude_unset: Exclude fields that were not explicitly set.
      exclude_defaults: Exclude fields that have their default value.
      exclude_none: Exclude fields that are None.
    
    Returns: The processed data, potentially with deferred building.
    
    Note: This is an experimental feature.
```

----------------------------------------

TITLE: Reuse TypeAdapter Instance
DESCRIPTION: Instantiating `TypeAdapter` repeatedly within a function can lead to performance degradation as new validators and serializers are constructed each time. Reusing a single instance of `TypeAdapter` is more efficient.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/performance.md#_snippet_0

LANGUAGE: python
CODE:
```
from pydantic import TypeAdapter


# Bad practice: Instantiating adapter inside the function
def my_func_bad():
    adapter = TypeAdapter(list[int])
    # do something with adapter


# Good practice: Instantiate adapter once and reuse
adapter_good = TypeAdapter(list[int])

def my_func_good():
    # do something with adapter_good
```

----------------------------------------

TITLE: Handle less_than Error in Pydantic
DESCRIPTION: This error is raised when an input value fails the `lt` (less than) constraint defined using `Field` in Pydantic. The example sets a field `x` to require a value less than 10, and passing 10 triggers the 'less_than' validation error.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_53

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field, ValidationError


class Model(BaseModel):
    x: int = Field(lt=10)


try:
    Model(x=10)
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'less_than'
```

----------------------------------------

TITLE: Mypy Plugin for BaseModel.__init__
DESCRIPTION: Introduces a mypy plugin to improve type checking for `BaseModel.__init__` and other related aspects. This enhances static analysis capabilities for Pydantic models.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_253

LANGUAGE: APIDOC
CODE:
```
# To enable the mypy plugin, add the following to your mypy.ini or setup.cfg:
# [mypy]
# plugins = pydantic.mypy

# Example of what the plugin helps check:
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str

# The plugin helps catch type errors in initialization:
# user = User(id='1', name='Alice') # Mypy would flag 'id' type mismatch

```

----------------------------------------

TITLE: Demonstrate dataclass_type error in Pydantic
DESCRIPTION: This example demonstrates the `dataclass_type` error, which is raised when a value is not valid for a dataclass field. It defines nested dataclasses and shows that passing an incorrect type (an integer instead of an `Inner` dataclass instance) for a nested field triggers this validation error.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_12

LANGUAGE: python
CODE:
```
from pydantic import ValidationError, dataclasses


@dataclasses.dataclass
class Inner:
    x: int


@dataclasses.dataclass
class Outer:
    y: Inner


Outer(y=Inner(x=1))  # OK

try:
    Outer(y=1)
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'dataclass_type'
```

----------------------------------------

TITLE: Pydantic: Support typing.Annotated for Field hints
DESCRIPTION: Demonstrates how to use `typing.Annotated` to specify Pydantic `Field` configurations directly within type hints. Other annotations are ignored but preserved.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_169

LANGUAGE: python
CODE:
```
from typing import Annotated
from pydantic import BaseModel, Field

class MyModel(BaseModel):
    my_field: Annotated[int, Field(gt=0, description="A positive integer")]

# get_type_hints(MyModel, include_extras=True) would show the Field details.
```

----------------------------------------

TITLE: Fetch and Validate Single User with httpx and Pydantic
DESCRIPTION: Fetches user data from the JSONPlaceholder API using httpx and validates the response against a Pydantic BaseModel. It demonstrates basic model creation and data retrieval for a single record.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/examples/requests.md#_snippet_0

LANGUAGE: python
CODE:
```
import httpx

from pydantic import BaseModel, EmailStr


class User(BaseModel):
    id: int
    name: str
    email: EmailStr


url = 'https://jsonplaceholder.typicode.com/users/1'

response = httpx.get(url)
response.raise_for_status()

user = User.model_validate(response.json())
print(repr(user))
#> User(id=1, name='Leanne Graham', email='Sincere@april.biz')
```

----------------------------------------

TITLE: Handle less_than_equal Error in Pydantic
DESCRIPTION: This error is raised when an input value fails the `le` (less than or equal to) constraint defined using `Field` in Pydantic. The example sets a field `x` to require a value less than or equal to 10, and passing 11 triggers the 'less_than_equal' validation error.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_54

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field, ValidationError


class Model(BaseModel):
    x: int = Field(le=10)


try:
    Model(x=11)
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'less_than_equal'
```

----------------------------------------

TITLE: Python Version Support and Dependencies
DESCRIPTION: Information on supported Python versions and changes related to code formatting and dependencies.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_299

LANGUAGE: APIDOC
CODE:
```
Python Support:
  - Supported Python 3.7.

Code Formatting:
  - Moved codebase to use black for formatting.

Dependency Changes:
  - Breaking change: Removed msgpack parsing.
```

----------------------------------------

TITLE: Structural Pattern Matching on BaseModel
DESCRIPTION: Adds support and documentation for structural pattern matching (PEP 636) on Pydantic's `BaseModel`, enabling more expressive data handling.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_88

LANGUAGE: python
CODE:
```
from pydantic import BaseModel

class Point(BaseModel):
    x: int
    y: int

def process_point(p):
    match p:
        case Point(x=0, y=0):
            print("Origin")
        case Point(x=x_val, y=y_val):
            print(f"Point at ({x_val}, {y_val})")

# Example:
# process_point(Point(x=1, y=2))
```

----------------------------------------

TITLE: Configure Strict Mode with @validate_call in Pydantic
DESCRIPTION: Illustrates enabling strict mode for function calls using the @validate_call decorator. By passing config=ConfigDict(strict=True) to the decorator, Pydantic will strictly validate function arguments against their type hints, preventing implicit type coercion. The example shows a ValidationError when a string '1' is passed to an integer parameter.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/strict_mode.md#_snippet_18

LANGUAGE: python
CODE:
```
from pydantic import ConfigDict, ValidationError, validate_call


@validate_call(config=ConfigDict(strict=True))
def foo(x: int) -> int:
    return x


try:
    foo('1')
except ValidationError as exc:
    print(exc)
    
```

----------------------------------------

TITLE: Demonstrate date_type error in Pydantic
DESCRIPTION: This example shows the `date_type` error, raised when the input value's type is invalid for a `date` field. It also covers strict fields where the input must be an instance of `date`. The snippet attempts to validate `None` for a `date` field, triggering the error.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_18

LANGUAGE: python
CODE:
```
from datetime import date

from pydantic import BaseModel, ValidationError


class Model(BaseModel):
    x: date


try:
    Model(x=None)
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'date_type'

# This error is also raised for strict fields when the input value is not an instance of `date`.
```

----------------------------------------

TITLE: Pydantic V2: Constrained Types Migration
DESCRIPTION: Illustrates the migration from Pydantic V1's Constrained types (like ConstrainedInt) to Pydantic V2's use of Annotated with Field for defining constraints. It shows the old and new syntax.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_32

LANGUAGE: python
CODE:
```
# Pydantic V1 syntax (removed in V2)
from pydantic import BaseModel, ConstrainedInt


class MyIntV1(ConstrainedInt):
    ge = 0


class ModelV1(BaseModel):
    x: MyIntV1
```

LANGUAGE: python
CODE:
```
# Pydantic V2 syntax
from typing import Annotated

from pydantic import BaseModel, Field

MyIntV2 = Annotated[int, Field(ge=0)]


class ModelV2(BaseModel):
    x: MyIntV2
```

----------------------------------------

TITLE: Exclude Invalid Fields from Pydantic JSON Schema
DESCRIPTION: This example illustrates how to prevent fields that do not have a valid JSON schema representation (e.g., callables) from being included in the final schema. This is achieved by overriding the `handle_invalid_for_json_schema` method to raise `PydanticOmit`, effectively omitting the field.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/json_schema.md#_snippet_31

LANGUAGE: python
CODE:
```
from typing import Callable

from pydantic_core import PydanticOmit, core_schema

from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue


class MyGenerateJsonSchema(GenerateJsonSchema):
    def handle_invalid_for_json_schema(
        self, schema: core_schema.CoreSchema, error_info: str
    ) -> JsonSchemaValue:
        raise PydanticOmit


def example_callable():
    return 1


class Example(BaseModel):
    name: str = 'example'
    function: Callable = example_callable


instance_example = Example()

validation_schema = instance_example.model_json_schema(
    schema_generator=MyGenerateJsonSchema, mode='validation'
)
print(validation_schema)
"""
{
    'properties': {
        'name': {'default': 'example', 'title': 'Name', 'type': 'string'}
    },
    'title': 'Example',
    'type': 'object',
}
"""
```

----------------------------------------

TITLE: Regex Anchoring Semantics Documentation
DESCRIPTION: Clarifies the default behavior of regular expression anchoring within Pydantic's validation. This documentation update helps users understand how regex patterns are applied to input strings.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_219

LANGUAGE: python
CODE:
```
# Documentation update regarding regex anchoring.
# Example: A regex like 'abc' might be implicitly anchored to match the whole string
# depending on Pydantic's internal handling, which is now documented.
```

----------------------------------------

TITLE: Basic Default Values
DESCRIPTION: Demonstrates setting default values for fields using direct assignment and the `Field` function with `default`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/fields.md#_snippet_6

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field


class User(BaseModel):
    # Both fields aren't required:
    name: str = 'John Doe'
    age: int = Field(default=20)
```

----------------------------------------

TITLE: Generate JSON Schema with Pydantic `computed_field`
DESCRIPTION: This example demonstrates how the `computed_field` decorator influences the JSON Schema generated for a Pydantic model. It shows that computed properties like `volume` are included in the schema as read-only fields, ensuring they are part of the model's contract during serialization.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/fields.md#_snippet_39

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, computed_field


class Box(BaseModel):
    width: float
    height: float
    depth: float

    @computed_field
    @property  # (1)!
    def volume(self) -> float:
        return self.width * self.height * self.depth


print(Box.model_json_schema(mode='serialization'))
"""
{
    'properties': {
        'width': {'title': 'Width', 'type': 'number'},
        'height': {'title': 'Height', 'type': 'number'},
        'depth': {'title': 'Depth', 'type': 'number'},
        'volume': {'readOnly': True, 'title': 'Volume', 'type': 'number'}
    },
    'required': ['width', 'height', 'depth', 'volume'],
    'title': 'Box',
    'type': 'object'
}
"""
```

----------------------------------------

TITLE: Pydantic v2 @field_validator with ValidationInfo
DESCRIPTION: Demonstrates the usage of Pydantic v2's @field_validator decorator, showing how to access validation context via the 'info' parameter, including 'config' and 'field_name'. This replaces the older 'config' and 'field' arguments available in v1.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_18

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ValidationInfo, field_validator


class Model(BaseModel):
    x: int

    @field_validator('x')
    def val_x(cls, v: int, info: ValidationInfo) -> int:
        assert info.config is not None
        print(info.config.get('title'))
        # Expected output: Model
        print(cls.model_fields[info.field_name].is_required())
        # Expected output: True
        return v


Model(x=1)
```

----------------------------------------

TITLE: Pydantic: Use WithJsonSchema to Override JSON Schema Generation
DESCRIPTION: The `WithJsonSchema` annotation allows overriding the generated JSON schema for a type without implementing schema generation methods. It requires providing the full schema, including the 'type', and is preferred over manual implementation for simplicity. This example demonstrates overriding the schema for an integer type.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/json_schema.md#_snippet_12

LANGUAGE: python
CODE:
```
import json
from typing import Annotated

from pydantic import BaseModel, WithJsonSchema

MyInt = Annotated[
    int,
    WithJsonSchema({'type': 'integer', 'examples': [1, 0, -1]}),
]


class Model(BaseModel):
    a: MyInt


print(json.dumps(Model.model_json_schema(), indent=2))
```

----------------------------------------

TITLE: Import Pydantic V1 Features (Universal)
DESCRIPTION: Provides a robust method to import Pydantic V1 features, compatible with any Pydantic version (V1 or V2). It uses a try-except block to fall back to the V1 namespace if direct import fails.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_7

LANGUAGE: python
CODE:
```
try:
    from pydantic.v1.fields import ModelField
except ImportError:
    from pydantic.fields import ModelField
```

----------------------------------------

TITLE: Handle int_parsing_size Error in Pydantic
DESCRIPTION: This error is raised when attempting to parse an integer from a string that exceeds the maximum range permitted by Python's str to int parsing. The example demonstrates triggering this error with both direct Python parsing and JSON parsing within Pydantic models.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/validation_errors.md#_snippet_44

LANGUAGE: python
CODE:
```
import json

from pydantic import BaseModel, ValidationError


class Model(BaseModel):
    x: int


# from Python
assert Model(x='1' * 4_300).x == int('1' * 4_300)  # OK

too_long = '1' * 4_301
try:
    Model(x=too_long)
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'int_parsing_size'

# from JSON
try:
    Model.model_validate_json(json.dumps({'x': too_long}))
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'int_parsing_size'
```

----------------------------------------

TITLE: Pydantic `.json()` Method Deprecation
DESCRIPTION: Highlights the deprecation of the `.json()` method in Pydantic V2, recommending the use of `model_dump_json()` instead. It also notes potential issues with arguments like `indent` and provides a workaround.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_14

LANGUAGE: APIDOC
CODE:
```
Pydantic `.json()` Method Deprecation:

- The `.json()` method for serializing models to JSON strings is deprecated in Pydantic V2.
- Recommended Replacement: Use `model_dump_json()`.

Reasons for Deprecation:
- Potential for confusing errors when using arguments like `indent` or `ensure_ascii`.
- `model_dump_json()` provides a more consistent and robust API.

Workaround for Arguments:
If you need to use arguments like `indent` or `ensure_ascii` with the deprecated `.json()` method, a workaround involves using `model_dump()` first and then passing the result to Python's standard `json` module.

Example Usage:
```python
import json
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float

item = Item(name='Foo', price=12.50)

# Recommended V2 approach:
json_output_v2 = item.model_dump_json(indent=2)
print(json_output_v2)

# Workaround for arguments if .json() must be used (not recommended):
# This is to illustrate the workaround, but model_dump_json is preferred.
# json_output_workaround = json.dumps(item.model_dump(), indent=2)
# print(json_output_workaround)
```
```

----------------------------------------

TITLE: Pydantic Wrap Serializer - Annotated Pattern
DESCRIPTION: Shows how to use WrapSerializer with the annotated pattern, allowing custom logic to be executed before or after Pydantic's default serialization by including a 'handler' parameter.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/serialization.md#_snippet_6

LANGUAGE: python
CODE:
```
from typing import Annotated, Any

from pydantic import BaseModel, SerializerFunctionWrapHandler, WrapSerializer


def ser_number(value: Any, handler: SerializerFunctionWrapHandler) -> int:
    return handler(value) + 1


class Model(BaseModel):
    number: Annotated[int, WrapSerializer(ser_number)]


print(Model(number=4).model_dump())
#> {'number': 5}
```

----------------------------------------

TITLE: Pydantic: Use SkipJsonSchema to Skip Fields in JSON Schema
DESCRIPTION: The `SkipJsonSchema` annotation is used to exclude a field or part of a field's specifications from the generated JSON schema. This is useful for selectively omitting certain data from schema outputs. Refer to the API documentation for detailed usage and examples.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/json_schema.md#_snippet_13

LANGUAGE: APIDOC
CODE:
```
pydantic.json_schema.SkipJsonSchema

Description:
  Used to skip an included field (or part of a field's specifications) from the generated JSON schema.

Usage:
  Apply the `SkipJsonSchema` annotation to a field or type to prevent its inclusion in the JSON schema.

Related:
  - `pydantic.json_schema.WithJsonSchema`: For overriding JSON schema generation.
  - Validators: For fine-tuning JSON schema of fields with validators using `json_schema_input_type`.
```

----------------------------------------

TITLE: Pydantic Inconsistency: Model Completion Status
DESCRIPTION: Highlights an inconsistency in Pydantic's backward compatibility logic where a model defined within a function, referencing other types, might incorrectly report its completion status. This example shows a dataclass with forward references to 'Model' and 'Inner', and a BaseModel 'Model' within a function, leading to an unexpected '__pydantic_complete__' status.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/internals/resolving_annotations.md#_snippet_6

LANGUAGE: python
CODE:
```
from dataclasses import dataclass

from pydantic import BaseModel


@dataclass
class Foo:
    # `a` and `b` shouldn't resolve:
    a: 'Model'
    b: 'Inner'


def func():
    Inner = int

    class Model(BaseModel):
        foo: Foo

    Model.__pydantic_complete__
    #> True, should be False.
```

----------------------------------------

TITLE: TypeAdapter: Validate and Generate Schema for List[int]
DESCRIPTION: Demonstrates creating a `TypeAdapter` for a `list[int]`, validating Python data against it, and generating its JSON schema. This showcases `TypeAdapter`'s ability to handle non-BaseModel types.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_27

LANGUAGE: python
CODE:
```
from pydantic import TypeAdapter

adapter = TypeAdapter(list[int])
assert adapter.validate_python(['1', '2', '3']) == [1, 2, 3]
print(adapter.json_schema())
#> {'items': {'type': 'integer'}, 'type': 'array'}
```

----------------------------------------

TITLE: Merging Parent and Child Configuration
DESCRIPTION: Shows how Pydantic merges configuration from parent and child models, prioritizing the child's configuration when conflicts arise.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/config.md#_snippet_7

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ConfigDict


class Parent(BaseModel):
    model_config = ConfigDict(extra='allow', str_to_lower=False)


class Model(Parent):
    model_config = ConfigDict(str_to_lower=True)

    x: str

m = Model(x='FOO', y='bar')
print(m.model_dump())
#> {'x': 'foo', 'y': 'bar'}
print(Model.model_config)
#> {'extra': 'allow', 'str_to_lower': True}

```

----------------------------------------

TITLE: Pydantic V2 Field Requirements and Nullability
DESCRIPTION: Illustrates Pydantic V2's updated logic for field requirements and nullability, which now more closely matches Python dataclasses. It clarifies how `Optional` and `Any` annotations behave and how default values impact whether a field is required.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_24

LANGUAGE: python
CODE:
```
from typing import Optional

from pydantic import BaseModel, ValidationError


class Foo(BaseModel):
    f1: str  # required, cannot be None
    f2: Optional[str]  # required, can be None - same as str | None
    f3: Optional[str] = None  # not required, can be None
    f4: str = 'Foobar'  # not required, but cannot be None


try:
    # Example demonstrating validation errors for f1 being None
    Foo(f1=None, f2=None, f4='b')
except ValidationError as e:
    print(e)
    # Output will show validation error for f1
    # 1 validation error for Foo
    # f1
    #   Input should be a valid string [type=string_type, input_value=None, input_type=NoneType]
```

----------------------------------------

TITLE: Partial JSON List Parsing with pydantic_core.from_json
DESCRIPTION: Demonstrates parsing incomplete JSON lists using `pydantic_core.from_json`. Shows how `allow_partial=False` raises an error for malformed JSON, while `allow_partial=True` deserializes the valid portion.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/json.md#_snippet_1

LANGUAGE: python
CODE:
```
from pydantic_core import from_json

partial_json_data = '["aa", "bb", "c'

try:
    result = from_json(partial_json_data, allow_partial=False)
except ValueError as e:
    print(e)
    #> EOF while parsing a string at line 1 column 15

result = from_json(partial_json_data, allow_partial=True)
print(result)
#> ['aa', 'bb']
```

----------------------------------------

TITLE: Pydantic V2 model_dump_json() Compaction and Separators
DESCRIPTION: Illustrates that Pydantic V2's `model_dump_json()` output is compacted for space efficiency and may differ from standard `json.dumps()`. It shows how to match `json.dumps()` output by adjusting separators.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_16

LANGUAGE: python
CODE:
```
import json

from pydantic import BaseModel as V2BaseModel
from pydantic.v1 import BaseModel as V1BaseModel


class V1Model(V1BaseModel):
    a: list[str]


class V2Model(V2BaseModel):
    a: list[str]


v1_model = V1Model(a=['fancy', 'sushi'])
v2_model = V2Model(a=['fancy', 'sushi'])

# V1
print(v1_model.json())
#> {"a": ["fancy", "sushi"]}

# V2
print(v2_model.model_dump_json())
#> {"a":["fancy","sushi"]}

# Plain json.dumps
print(json.dumps(v2_model.model_dump()))
#> {"a": ["fancy", "sushi"]}

# Modified json.dumps
print(json.dumps(v2_model.model_dump(), separators=(',', ':')))
#> {"a":["fancy","sushi"]}
```

----------------------------------------

TITLE: Pydantic JSON Serialization API Methods
DESCRIPTION: Lists key API methods available in Pydantic for serializing data to JSON format, providing options for different use cases like BaseModel instances or TypeAdapter usage.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/json.md#_snippet_6

LANGUAGE: APIDOC
CODE:
```
pydantic.main.BaseModel.model_dump_json
pydantic.type_adapter.TypeAdapter.dump_json
pydantic_core.to_json
```

----------------------------------------

TITLE: Accept empty query/fragment URL parts
DESCRIPTION: Allows URL parsing to accept empty query or fragment parts. This improves flexibility when dealing with URLs that may omit these components.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_126

LANGUAGE: python
CODE:
```
Accept empty query/fragment URL parts, [#1807](https://github.com/pydantic/pydantic/pull/1807) by @xavier
```

----------------------------------------

TITLE: Type Hints for `BaseSettings.Config`
DESCRIPTION: Adds type hints to the `Config` inner class of `pydantic.BaseSettings`. This improves static analysis and helps avoid mypy errors, ensuring better compatibility with type checking tools.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_117

LANGUAGE: python
CODE:
```
from pydantic import BaseSettings

class Settings(BaseSettings):
    class Config:
        # Type hints added here for better mypy compatibility
        env_file: str | None = None
        env_file_encoding: str | None = None
        env_prefix: str = ''
        case_sensitive: bool = False
        extra: str | None = None
        # ... other Config attributes
```

----------------------------------------

TITLE: Pydantic BeforeValidator with Annotated Pattern
DESCRIPTION: Demonstrates using `BeforeValidator` with `Annotated` to preprocess input before Pydantic's core validation. It shows how to ensure a value is a list, handling cases where a single item is provided, and how Pydantic still validates the processed item.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/validators.md#_snippet_4

LANGUAGE: python
CODE:
```
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ValidationError


def ensure_list(value: Any) -> Any:  # (1)!
    if not isinstance(value, list):  # (2)!
        return [value]
    else:
        return value


class Model(BaseModel):
    numbers: Annotated[list[int], BeforeValidator(ensure_list)]


print(Model(numbers=2))
#> numbers=[2]
try:
    Model(numbers='str')
except ValidationError as err:
    print(err)  # (3)!
    """
    1 validation error for Model
    numbers.0
      Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='str', input_type=str]
    """

```

----------------------------------------

TITLE: RootModel Subclass with Custom Methods
DESCRIPTION: Demonstrates creating a subclass of a parametrized `RootModel` and adding custom methods for enhanced functionality.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_49

LANGUAGE: python
CODE:
```
from pydantic import RootModel


class Pets(RootModel[list[str]]):
    def describe(self) -> str:
        return f'Pets: {", ".join(self.root)}'


my_pets = Pets.model_validate(['dog', 'cat'])

print(my_pets.describe())
#> Pets: dog, cat
```

----------------------------------------

TITLE: Adding Validators with create_model and field_validator
DESCRIPTION: Demonstrates how to dynamically create a model with custom validators using `create_model` and the `field_validator` decorator.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_46

LANGUAGE: python
CODE:
```
from pydantic import ValidationError, create_model, field_validator


def alphanum(cls, v):
    assert v.isalnum(), 'must be alphanumeric'
    return v


validators = {
    'username_validator': field_validator('username')(alphanum)  # (1)!
}

UserModel = create_model(
    'UserModel', username=(str, ...), __validators__=validators
)

user = UserModel(username='scolvin')
print(user)
#> username='scolvin'

try:
    UserModel(username='scolvi%n')
except ValidationError as e:
    print(e)
    """
    1 validation error for UserModel
    username
      Assertion failed, must be alphanumeric [type=assertion_error, input_value='scolvi%n', input_type=str]
    """

```

----------------------------------------

TITLE: Pydantic V1 to V2 Deprecated Features
DESCRIPTION: This API documentation maps deprecated Pydantic V1 features to their corresponding locations or replacements in Pydantic V2. It serves as a reference for users updating their projects.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_35

LANGUAGE: APIDOC
CODE:
```
APIDOC: Pydantic V1 to V2 Migration Guide

Description:
  This guide details the migration path for deprecated Pydantic V1 features to Pydantic V2. It provides a direct mapping from V1 module/function paths to their V2 equivalents or new locations.

Mappings:
  - pydantic.tools.schema_of -> pydantic.deprecated.tools.schema_of
  - pydantic.tools.parse_obj_as -> pydantic.deprecated.tools.parse_obj_as
  - pydantic.tools.schema_json_of -> pydantic.deprecated.tools.schema_json_of
  - pydantic.json.pydantic_encoder -> pydantic.deprecated.json.pydantic_encoder
  - pydantic.validate_arguments -> pydantic.deprecated.decorator.validate_arguments
  - pydantic.json.custom_pydantic_encoder -> pydantic.deprecated.json.custom_pydantic_encoder
  - pydantic.json.ENCODERS_BY_TYPE -> pydantic.deprecated.json.ENCODERS_BY_TYPE
  - pydantic.json.timedelta_isoformat -> pydantic.deprecated.json.timedelta_isoformat
  - pydantic.decorator.validate_arguments -> pydantic.deprecated.decorator.validate_arguments
  - pydantic.class_validators.validator -> pydantic.deprecated.class_validators.validator
  - pydantic.class_validators.root_validator -> pydantic.deprecated.class_validators.root_validator
  - pydantic.utils.deep_update -> pydantic.v1.utils.deep_update
  - pydantic.utils.GetterDict -> pydantic.v1.utils.GetterDict
  - pydantic.utils.lenient_issubclass -> pydantic.v1.utils.lenient_issubclass
  - pydantic.utils.lenient_isinstance -> pydantic.v1.utils.lenient_isinstance
  - pydantic.utils.is_valid_field -> pydantic.v1.utils.is_valid_field
  - pydantic.utils.update_not_none -> pydantic.v1.utils.update_not_none
  - pydantic.utils.import_string -> pydantic.v1.utils.import_string
  - pydantic.utils.Representation -> pydantic.v1.utils.Representation
  - pydantic.utils.ROOT_KEY -> pydantic.v1.utils.ROOT_KEY
  - pydantic.utils.smart_deepcopy -> pydantic.v1.utils.smart_deepcopy
  - pydantic.utils.sequence_like -> pydantic.v1.utils.sequence_like

Notes:
  - Features marked as 'deprecated' in V1 are often moved to a 'deprecated' submodule in V2.
  - Some utilities are moved to a 'v1' submodule in V2 to maintain backward compatibility or indicate V1-specific behavior.
```

----------------------------------------

TITLE: Pydantic BaseSettings Secret Files
DESCRIPTION: Introduces the ability for `BaseSettings` models to read sensitive configuration values from secret files.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_202

LANGUAGE: python
CODE:
```
from pydantic import BaseSettings

class Settings(BaseSettings):
    api_key: str

    class Config:
        # Example: Specify a file to load secrets from
        # secrets_dir = '/path/to/secrets'
        # api_key will be loaded from a file named 'api_key' in secrets_dir
        pass

# settings = Settings()
# print(settings.api_key)
```

----------------------------------------

TITLE: SerializeAsAny Runtime Setting
DESCRIPTION: Illustrates the use of the `serialize_as_any` runtime setting in Pydantic's serialization methods. Setting it to True enables duck typed serialization, similar to V1 behavior, while False uses V2 defaults.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/serialization.md#_snippet_18

LANGUAGE: python
CODE:
```
from pydantic import BaseModel


class User(BaseModel):
    name: str


class UserLogin(User):
    password: str


class OuterModel(BaseModel):
    user1: User
    user2: User


user = UserLogin(name='pydantic', password='password')

outer_model = OuterModel(user1=user, user2=user)
print(outer_model.model_dump(serialize_as_any=True))  # (1)!

print(outer_model.model_dump(serialize_as_any=False))  # (2)!
```

----------------------------------------

TITLE: Pydantic API Documentation
DESCRIPTION: API reference for Pydantic serialization functions, including @field_serializer and @model_serializer. Details their usage, parameters, and modes ('plain', 'wrap').

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/serialization.md#_snippet_13

LANGUAGE: APIDOC
CODE:
```
pydantic.functional_serializers.model_serializer
  Serializes an entire model.
  Modes:
    - 'plain': Called unconditionally. Can return any type.
    - 'wrap': Takes a 'handler' argument for custom pre/post processing.

pydantic.field_serializer
  Decorator to apply serialization to specific fields.
  Arguments:
    *field_names: Names of fields to apply the serializer to. Use '*' for all fields.
    mode: 'plain' or 'wrap'.
    check_fields: Boolean, whether to check if fields exist on the model (default True).

SerializerFunctionWrapHandler
  A callable passed to 'wrap' mode serializers. It handles the default serialization process.
```

----------------------------------------

TITLE: Pydantic Web Template Inheritance
DESCRIPTION: This snippet illustrates the Jinja2 template inheritance pattern used in the Pydantic project. It extends a base template and includes specific content blocks and partials for announcements and main content.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/theme/main.html#_snippet_0

LANGUAGE: jinja
CODE:
```
{% extends "base.html" %}
{% block announce %}
  {% include 'announce.html' ignore missing %}
{% endblock %}
{% block content %}
  {{ super() }}
  {% include 'mkdocs\_run_deps.html' ignore missing %}
{% endblock %}
```

----------------------------------------

TITLE: Define Pydantic Model with Type Hints
DESCRIPTION: Demonstrates defining a Pydantic `BaseModel` using Python type hints for data validation. It shows basic types, `Literal`, `Annotated` with constraints, and complex nested types for robust data structuring.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/why.md#_snippet_0

LANGUAGE: python
CODE:
```
from typing import Annotated, Literal

from annotated_types import Gt

from pydantic import BaseModel


class Fruit(BaseModel):
    name: str  # (1)!
    color: Literal['red', 'green']  # (2)!
    weight: Annotated[float, Gt(0)]  # (3)!
    bazam: dict[str, list[tuple[int, bool, float]]]  # (4)!


print(
    Fruit(
        name='Apple',
        color='red',
        weight=4.2,
        bazam={'foobar': [(1, True, 0.1)]},
    )
)
#> name='Apple' color='red' weight=4.2 bazam={'foobar': [(1, True, 0.1)]}
```

----------------------------------------

TITLE: Create New Feature Branch
DESCRIPTION: Steps to create a new Git branch for your contributions and make your changes. This ensures that your work is isolated and can be easily tracked.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/contributing.md#_snippet_5

LANGUAGE: bash
CODE:
```
git checkout -b my-new-feature-branch
# Make your changes...
```

----------------------------------------

TITLE: Pydantic Model Creation and Validation
DESCRIPTION: Covers methods and features related to creating and validating Pydantic models. This includes dynamic model creation and advanced validator usage.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_302

LANGUAGE: APIDOC
CODE:
```
Pydantic Model Creation and Validation:

create_model(name, **field_definitions)
  - Dynamically creates a Pydantic model.

@validate('*')
  - Wildcard validator decorator that applies to all fields.

@validator('field_name', always=True)
  - Validator decorator that can be configured to run always.
```

----------------------------------------

TITLE: Python Mode Serialization with model_dump()
DESCRIPTION: Demonstrates serializing Pydantic models to Python dictionaries using `model_dump()`. Shows default output and output with `by_alias=True`. Also illustrates converting to JSON-compatible types with `mode='json'`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/serialization.md#_snippet_0

LANGUAGE: python
CODE:
```
from typing import Optional

from pydantic import BaseModel, Field


class BarModel(BaseModel):
    whatever: tuple[int, ...]


class FooBarModel(BaseModel):
    banana: Optional[float] = 1.1
    foo: str = Field(serialization_alias='foo_alias')
    bar: BarModel


m = FooBarModel(banana=3.14, foo='hello', bar={'whatever': (1, 2)})

# returns a dictionary:
print(m.model_dump())
#> {'banana': 3.14, 'foo': 'hello', 'bar': {'whatever': (1, 2)}}

print(m.model_dump(by_alias=True))
#> {'banana': 3.14, 'foo_alias': 'hello', 'bar': {'whatever': (1, 2)}}

print(m.model_dump(mode='json'))
#> {'banana': 3.14, 'foo': 'hello', 'bar': {'whatever': [1, 2]}}
```

----------------------------------------

TITLE: Pydantic Config Options
DESCRIPTION: Details various configuration options available within Pydantic models, such as aliasing, error message templating, and whitespace stripping. These settings control model behavior and validation.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_300

LANGUAGE: APIDOC
CODE:
```
Pydantic Configuration:

Config.allow_population_by_alias: bool
  - Allows population of model fields using their aliases.

Config.error_msg_templates: dict
  - Defines custom error message templates for validation errors.

Config.anystr_strip_whitespace: bool
  - Controls whether whitespace is automatically stripped from string fields.

constr(strip_whitespace=True): kwarg
  - Option for constrained string types to strip whitespace.
```

----------------------------------------

TITLE: Pydantic BaseModel Construct and Default Factory
DESCRIPTION: Introduces support for `default_factory` with `BaseModel.construct` and deprecates `__field_defaults__`. Users should now use the `.get_default()` method on fields within the `__fields__` attribute.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_195

LANGUAGE: python
CODE:
```
class MyModel:
    # ...
    # Previously: __field_defaults__ = {...}
    # Now: use default_factory with construct
    # Access defaults via .get_default() on fields in __fields__
```

----------------------------------------

TITLE: Using '__all__' for Sequence Exclusion in Pydantic
DESCRIPTION: Demonstrates using the special '__all__' key with `exclude` to apply an exclusion pattern to all items within a sequence (list) in Pydantic models.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/serialization.md#_snippet_23

LANGUAGE: python
CODE:
```
from pydantic import BaseModel


class Hobby(BaseModel):
    name: str
    info: str


class User(BaseModel):
    hobbies: list[Hobby]


user = User(
    hobbies=[
        Hobby(name='Programming', info='Writing code and stuff'),
        Hobby(name='Gaming', info='Hell Yeah!!!'),
    ],
)

print(user.model_dump(exclude={'hobbies': {'__all__': {'info'}}}))
#> {'hobbies': [{'name': 'Programming'}, {'name': 'Gaming'}]}
```

----------------------------------------

TITLE: Pydantic Handling of `pathlib.Path`
DESCRIPTION: Describes Pydantic's validation for `pathlib.Path` types, which involves simply passing the input value directly to the `Path(v)` constructor for instantiation and validation.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/standard_library_types.md#_snippet_35

LANGUAGE: APIDOC
CODE:
```
pathlib.Path:
  Behavior: Simply uses the type itself for validation by passing the value to `Path(v)`.
```

----------------------------------------

TITLE: Pydantic Model Structural Pattern Matching (Python 3.10+)
DESCRIPTION: Shows how Pydantic models support structural pattern matching (PEP 636) in Python 3.10+, allowing for concise matching against model attributes and extraction of values. This provides a clean syntax for handling different model states or types.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_60

LANGUAGE: python
CODE:
```
from pydantic import BaseModel


class Pet(BaseModel):
    name: str
    species: str


a = Pet(name='Bones', species='dog')

match a:
    # match `species` to 'dog', declare and initialize `dog_name`
    case Pet(species='dog', name=dog_name):
        print(f'{dog_name} is a dog')
#> Bones is a dog
    # default case
    case _:
        print('No dog matched')
```

----------------------------------------

TITLE: ConfigDict validate_by_alias
DESCRIPTION: Shows how `ConfigDict.validate_by_alias=True` (default) allows validation using field aliases. Demonstrates validation with `my_alias`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/alias.md#_snippet_6

LANGUAGE: Python
CODE:
```
from pydantic import BaseModel, ConfigDict, Field


class Model(BaseModel):
    my_field: str = Field(validation_alias='my_alias')

    model_config = ConfigDict(validate_by_alias=True, validate_by_name=False)


print(repr(Model(my_alias='foo')))  # (1)!
#> Model(my_field='foo')
```

----------------------------------------

TITLE: Pydantic Model Validator (Before Mode)
DESCRIPTION: Demonstrates a 'before' model validator in Pydantic, which runs prior to model instantiation. These validators receive raw input data and can transform or validate it before Pydantic's core validation process begins.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/validators.md#_snippet_13

LANGUAGE: python
CODE:
```
from typing import Any

from pydantic import BaseModel, model_validator


class UserModel(BaseModel):
    username: str

    @model_validator(mode='before')
    @classmethod
    def check_card_number_not_present(cls, data: Any) -> Any:  # (1)!
        if isinstance(data, dict):  # (2)!
            if 'card_number' in data:
                raise ValueError("'card_number' should not be included")
        return data
```

----------------------------------------

TITLE: Pydantic Model Signature with Custom Init
DESCRIPTION: Shows how Pydantic incorporates custom `__init__` methods into the generated model signature, ensuring that parameters defined in the custom initializer are correctly reflected. This allows for more control over model instantiation.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_57

LANGUAGE: python
CODE:
```
import inspect

from pydantic import BaseModel


class MyModel(BaseModel):
    id: int
    info: str = 'Foo'

    def __init__(self, id: int = 1, *, bar: str, **data) -> None:
        """My custom init!"""
        super().__init__(id=id, bar=bar, **data)


print(inspect.signature(MyModel))
#> (id: int = 1, *, bar: str, info: str = 'Foo') -> None
```

----------------------------------------

TITLE: Pydantic Dataclass Configuration
DESCRIPTION: Illustrates configuring Pydantic dataclasses using the `@dataclass` decorator with `ConfigDict` to set `str_max_length` and `validate_assignment=True`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/config.md#_snippet_2

LANGUAGE: python
CODE:
```
from pydantic import ConfigDict, ValidationError
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(str_max_length=10, validate_assignment=True))
class User:
    name: str


user = User(name='John Doe')
try:
    user.name = 'x' * 20
except ValidationError as e:
    print(e)
    """
    1 validation error for User
    name
      String should have at most 10 characters [type=string_too_long, input_value='xxxxxxxxxxxxxxxxxxxx', input_type=str]
    """

```

----------------------------------------

TITLE: Asserting Type Hint Correctness
DESCRIPTION: Demonstrates how to use `typing.assert_type` to verify type hint correctness for Pydantic constructs like `TypeAdapter`. This is crucial for ensuring static analysis tools correctly interpret Pydantic types.

SOURCE: https://github.com/pydantic/pydantic/blob/main/tests/typechecking/README.md#_snippet_0

LANGUAGE: Python
CODE:
```
from typing_extensions import assert_type

from pydantic import TypeAdapter

ta1 = TypeAdapter(int)
assert_type(ta1, TypeAdapter[int])
```

----------------------------------------

TITLE: Secrets Files Support in BaseSettings
DESCRIPTION: Adds support for loading sensitive configuration values from 'secrets files' when using `BaseSettings`. This provides a secure way to manage credentials and other secrets, separating them from code and environment variables.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_193

LANGUAGE: python
CODE:
```
# BaseSettings secrets files support
```

----------------------------------------

TITLE: Pydantic Alias Generators API
DESCRIPTION: API reference for Pydantic's alias generation capabilities, focusing on how to customize field name transformations. Includes options for root heading display.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/config.md#_snippet_1

LANGUAGE: APIDOC
CODE:
```
pydantic.alias_generators:
  options:
    show_root_heading: true
```

----------------------------------------

TITLE: Extending Base Model with Extra Fields via create_model
DESCRIPTION: Shows how to create a new model that inherits from an existing Pydantic `BaseModel` by using `create_model` and specifying `__base__`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_45

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, create_model


class FooModel(BaseModel):
    foo: str
    bar: int = 123


BarModel = create_model(
    'BarModel',
    apple=(str, 'russet'),
    banana=(str, 'yellow'),
    __base__=FooModel,
)
print(BarModel)
#> <class '__main__.BarModel'>
print(BarModel.model_fields.keys())
#> dict_keys(['foo', 'bar', 'apple', 'banana'])
```

----------------------------------------

TITLE: FailFast Annotation for Sequence Validation
DESCRIPTION: Demonstrates the `FailFast` annotation in Pydantic v2.8+ for sequence types. This feature allows validation to stop as soon as the first item in a sequence fails, improving performance at the cost of reduced error visibility for subsequent items. It requires Pydantic and `typing`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/performance.md#_snippet_5

LANGUAGE: python
CODE:
```
from typing import Annotated

from pydantic import FailFast, TypeAdapter, ValidationError

ta = TypeAdapter(Annotated[list[bool], FailFast()])
try:
    ta.validate_python([True, 'invalid', False, 'also invalid'])
except ValidationError as exc:
    print(exc)
    """
    1 validation error for list[bool]
    1
      Input should be a valid boolean, unable to interpret input [type=bool_parsing, input_value='invalid', input_type=str]
    """

```

----------------------------------------

TITLE: Pydantic create_model API Reference
DESCRIPTION: A utility function for dynamically creating Pydantic models at runtime. It allows for the creation of models without defining them as classes beforehand, useful for flexible data handling.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/base_model.md#_snippet_1

LANGUAGE: APIDOC
CODE:
```
pydantic.create_model
  options:
    show_root_heading: true
```

----------------------------------------

TITLE: Pydantic Model from SQLAlchemy Instance
DESCRIPTION: Demonstrates creating a Pydantic model from a SQLAlchemy ORM instance by setting `ConfigDict(from_attributes=True)`. This allows Pydantic to read attributes directly from the ORM object.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_17

LANGUAGE: python
CODE:
```
from typing import Annotated

from sqlalchemy import ARRAY, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from pydantic import BaseModel, ConfigDict, StringConstraints


class Base(DeclarativeBase):
    pass


class CompanyOrm(Base):
    __tablename__ = 'companies'

    id: Mapped[int] = mapped_column(primary_key=True, nullable=False)
    public_key: Mapped[str] = mapped_column(
        String(20), index=True, nullable=False, unique=True
    )
    domains: Mapped[list[str]] = mapped_column(ARRAY(String(255)))


class CompanyModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    public_key: Annotated[str, StringConstraints(max_length=20)]
    domains: list[Annotated[str, StringConstraints(max_length=255)]]


co_orm = CompanyOrm(
    id=123,
    public_key='foobar',
    domains=['example.com', 'foobar.com'],
)
print(co_orm)
# > <__main__.CompanyOrm object at 0x0123456789ab>
co_model = CompanyModel.model_validate(co_orm)
print(co_model)
# > id=123 public_key='foobar' domains=['example.com', 'foobar.com']

```

----------------------------------------

TITLE: Packaging Updates
DESCRIPTION: Details on dependency version bumps and build optimizations for Pydantic and pydantic-core. These updates aim to improve build times and leverage performance enhancements.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_14

LANGUAGE: bash
CODE:
```
Bump `ruff` from 0.9.2 to 0.9.5
Bump `pydantic-core` to v2.29.0
Use locally-built rust with symbols & pgo
```

----------------------------------------

TITLE: Pydantic ValidationError Handling
DESCRIPTION: Demonstrates how to define Pydantic models, apply field validators, and catch/process ValidationError exceptions. It shows how to print the exception directly or access structured error details.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/errors.md#_snippet_0

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field, ValidationError, field_validator


class Location(BaseModel):
    lat: float = 0.1
    lng: float = 10.1


class Model(BaseModel):
    is_required: float
    gt_int: int = Field(gt=42)
    list_of_ints: list[int]
    a_float: float
    recursive_model: Location

    @field_validator('a_float', mode='after')
    @classmethod
    def validate_float(cls, value: float) -> float:
        if value > 2.0:
            raise ValueError('Invalid float value')
        return value


data = {
    'list_of_ints': ['1', 2, 'bad'],
    'a_float': 3.0,
    'recursive_model': {'lat': 4.2, 'lng': 'New York'},
    'gt_int': 21,
}

try:
    Model(**data)
except ValidationError as e:
    print(e)
    # Expected output:
    # 5 validation errors for Model
    # is_required
    #   Field required [type=missing, input_value={'list_of_ints': ['1', 2,...ew York'}, 'gt_int': 21}, input_type=dict]
    # gt_int
    #   Input should be greater than 42 [type=greater_than, input_value=21, input_type=int]
    # list_of_ints.2
    #   Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='bad', input_type=str]
    # a_float
    #   Value error, Invalid float value [type=value_error, input_value=3.0, input_type=float]
    # recursive_model.lng
    #   Input should be a valid number, unable to parse string as a number [type=float_parsing, input_value='New York', input_type=str]

try:
    Model(**data)
except ValidationError as e:
    print(e.errors())
    # Expected output:
    # [
    #     {
    #         'type': 'missing',
    #         'loc': ('is_required',),
    #         'msg': 'Field required',
    #         'input': {
    #             'list_of_ints': ['1', 2, 'bad'],
    #             'a_float': 3.0,
    #             'recursive_model': {'lat': 4.2, 'lng': 'New York'},
    #             'gt_int': 21,
    #         },
    # ... (rest of the errors)
    # ]

```

----------------------------------------

TITLE: Pydantic Badges in reStructuredText
DESCRIPTION: Provides the syntax for including Pydantic version badges within reStructuredText documentation. These badges serve as visual indicators and links to the Pydantic project.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/contributing.md#_snippet_13

LANGUAGE: rst
CODE:
```
.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v1.json
    :target: https://pydantic.dev
    :alt: Pydantic

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json
    :target: https://pydantic.dev
    :alt: Pydantic
```

----------------------------------------

TITLE: Pydantic: Default Values with Field (Python)
DESCRIPTION: Illustrates using `pydantic.Field` to provide default values for model attributes. It highlights the requirement for `default` to be a keyword argument for type checkers like Pyright to correctly infer optional fields, and explains a limitation related to runtime vs. type-checker behavior.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/integrations/visual_studio_code.md#_snippet_8

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field


class Knight(BaseModel):
    title: str = Field(default='Sir Lancelot')  # this is okay
    age: int = Field(
        23
    )  # this works fine at runtime but will case an error for pyright


lance = Knight()  # error: Argument missing for parameter "age"
```

----------------------------------------

TITLE: Generate Schema for Callable Arguments
DESCRIPTION: Shows how to use `generate_arguments_schema` to create a schema for a function's arguments. This schema can then be used with `SchemaValidator` to validate data (e.g., from JSON) into positional and keyword arguments for the function. The validated arguments can be bound to the function signature.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/experimental.md#_snippet_16

LANGUAGE: python
CODE:
```
from pydantic_core import SchemaValidator

from pydantic.experimental.arguments_schema import generate_arguments_schema


def func(p: bool, *args: str, **kwargs: int) -> None: ...


arguments_schema = generate_arguments_schema(func=func)

val = SchemaValidator(arguments_schema, config={'coerce_numbers_to_str': True})

args, kwargs = val.validate_json(
    '{"p": true, "args": ["arg1", 1], "kwargs": {"extra": 1}}'
)
print(args, kwargs)  # (1)!
#> (True, 'arg1', '1') {'extra': 1}
```

LANGUAGE: python
CODE:
```
from inspect import signature

# Assuming 'func', 'args', 'kwargs' are defined as above
signature(func).bind(*args, **kwargs).arguments
#> {'p': True, 'args': ('arg1', '1'), 'kwargs': {'extra': 1}}
```

----------------------------------------

TITLE: Configure Flake8 for Pydantic Linting
DESCRIPTION: Configures Flake8 to extend the ignored linting rules, specifically for Pydantic-related errors (PYDXXX codes). This allows you to customize which linting suggestions are displayed.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/integrations/linting.md#_snippet_1

LANGUAGE: ini
CODE:
```
[flake8]
extend-ignore = PYD001,PYD002
```

----------------------------------------

TITLE: Generated Pydantic Models from JSON Schema
DESCRIPTION: The Python code generated by datamodel-code-generator from the provided JSON Schema. It defines Pydantic models for 'Pet' and 'Person', including type hints, field descriptions, and constraints.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/integrations/datamodel_code_generator.md#_snippet_3

LANGUAGE: python
CODE:
```
# generated by datamodel-codegen:
#   filename:  person.json
#   timestamp: 2020-05-19T15:07:31+00:00
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, conint


class Pet(BaseModel):
    name: str | None = None
    age: int | None = None


class Person(BaseModel):
    first_name: str = Field(description="The person's first name.")
    last_name: str = Field(description="The person's last name.")
    age: conint(ge=0) | None = Field(None, description='Age in years.')
    pets: list[Pet] | None = None
    comment: Any | None = None
```

----------------------------------------

TITLE: Compare Pydantic Performance vs. Pure Python
DESCRIPTION: Benchmarks Pydantic's data validation performance against a pure Python implementation for parsing JSON and validating URLs. It highlights Pydantic's speed advantage, showing it to be significantly faster for common tasks.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/why.md#_snippet_1

LANGUAGE: python
CODE:
```
import json
import timeit
from urllib.parse import urlparse

import requests

from pydantic import HttpUrl, TypeAdapter

reps = 7
number = 100
r = requests.get('https://api.github.com/emojis')
r.raise_for_status()
emojis_json = r.content


def emojis_pure_python(raw_data):
    data = json.loads(raw_data)
    output = {}
    for key, value in data.items():
        assert isinstance(key, str)
        url = urlparse(value)
        assert url.scheme in ('https', 'http')
        output[key] = url


emojis_pure_python_times = timeit.repeat(
    'emojis_pure_python(emojis_json)',
    globals={
        'emojis_pure_python': emojis_pure_python,
        'emojis_json': emojis_json,
    },
    repeat=reps,
    number=number,
)
print(f'pure python: {min(emojis_pure_python_times) / number * 1000:0.2f}ms')
#> pure python: 5.32ms

type_adapter = TypeAdapter(dict[str, HttpUrl])
emojis_pydantic_times = timeit.repeat(
    'type_adapter.validate_json(emojis_json)',
    globals={
        'type_adapter': type_adapter,
        'HttpUrl': HttpUrl,
        'emojis_json': emojis_json,
    },
    repeat=reps,
    number=number,
)
print(f'pydantic: {min(emojis_pydantic_times) / number * 1000:0.2f}ms')
#> pydantic: 1.54ms

print(
    f'Pydantic {min(emojis_pure_python_times) / min(emojis_pydantic_times):0.2f}x faster'
)
#> Pydantic 3.45x faster

```

----------------------------------------

TITLE: Pydantic Dataclass Configuration
DESCRIPTION: Illustrates how to configure Pydantic dataclasses using `ConfigDict` either via the decorator argument or the `__pydantic_config__` attribute, specifically for `validate_assignment`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/dataclasses.md#_snippet_2

LANGUAGE: python
CODE:
```
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


# Option 1 -- using the decorator argument:
@dataclass(config=ConfigDict(validate_assignment=True))
class MyDataclass1:
    a: int


# Option 2 -- using an attribute:
@dataclass
class MyDataclass2:
    a: int

    __pydantic_config__ = ConfigDict(validate_assignment=True)
```

----------------------------------------

TITLE: Python: GenericModel with After Validator
DESCRIPTION: Demonstrates a generic Pydantic model with a custom validator. It shows how `model_validator(mode='after')` works on generic models, including output from validation and model printing.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_34

LANGUAGE: python
CODE:
```
from typing import Any, Generic, Self, TypeVar

from pydantic import BaseModel, model_validator

T = TypeVar('T')


class GenericModel(BaseModel, Generic[T]):
    a: T

    @model_validator(mode='after')
    def validate_after(self: Self) -> Self:
        print('after validator running custom validation...')
        return self


class Model(BaseModel):
    inner: GenericModel[Any]


m = Model.model_validate(Model(inner=GenericModel[int](a=1)))
print(repr(m))
```

----------------------------------------

TITLE: Dataclass Init=False and Extra=Allow Incompatibility in Pydantic
DESCRIPTION: Pydantic disallows the combination of `extra='allow'` with fields set to `init=False` on a dataclass. This prevents potential conflicts during schema building.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/usage_errors.md#_snippet_46

LANGUAGE: python
CODE:
```
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(extra='allow'))
class A:
    a: int = Field(init=False, default=1)
```

----------------------------------------

TITLE: Pydantic Model Parsing with `Union` and `Literal` for Type Specificity
DESCRIPTION: Shows how to order types in a `typing.Union` with `typing.Literal` to parse data into the most specific matching Pydantic model. Demonstrates how Pydantic resolves types based on the order in the union and literal values, allowing for hierarchical parsing.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/standard_library_types.md#_snippet_30

LANGUAGE: python
CODE:
```
from typing import Literal, Optional, Union

from pydantic import BaseModel


class Dessert(BaseModel):
    kind: str


class Pie(Dessert):
    kind: Literal['pie']
    flavor: Optional[str]


class ApplePie(Pie):
    flavor: Literal['apple']


class PumpkinPie(Pie):
    flavor: Literal['pumpkin']


class Meal(BaseModel):
    dessert: Union[ApplePie, PumpkinPie, Pie, Dessert]


print(type(Meal(dessert={'kind': 'pie', 'flavor': 'apple'}).dessert).__name__)
#> ApplePie
print(type(Meal(dessert={'kind': 'pie', 'flavor': 'pumpkin'}).dessert).__name__)
#> PumpkinPie
print(type(Meal(dessert={'kind': 'pie'}).dessert).__name__)
#> Dessert
print(type(Meal(dessert={'kind': 'cake'}).dessert).__name__)
#> Dessert
```

----------------------------------------

TITLE: Pydantic validate_call API Reference
DESCRIPTION: Provides details on the pydantic.validate_call decorator, its purpose, and how it leverages type annotations for argument validation and coercion.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/validation_decorator.md#_snippet_1

LANGUAGE: APIDOC
CODE:
```
pydantic.validate_call

Decorator to validate function arguments using Pydantic models and type annotations.

Purpose:
  Allows arguments passed to a function to be parsed and validated using the function's annotations before the function is called. It simplifies applying validation with minimal boilerplate.

Usage:
  Apply the decorator directly above the function definition.

  @validate_call
  def my_function(arg1: str, arg2: int):
      # function body
      pass

Parameter Types:
  - Parameter types are inferred from type annotations on the function.
  - If a parameter is not annotated, it defaults to `typing.Any`.
  - Supports all Pydantic-compatible types, including Pydantic models and custom types.

Type Coercion:
  - By default, types are coerced before being passed to the function.
  - For example, a string input for a `datetime.date` annotated parameter will be automatically converted to a `date` object.
  - This behavior can be controlled, for instance, by enabling strict mode.

Return Value Validation:
  - By default, the return value of the function is NOT validated.
  - To enable return value validation, set `validate_return=True` when applying the decorator:
    `@validate_call(validate_return=True)`

Dependencies:
  - Requires Pydantic library.

Error Handling:
  - Raises `pydantic.ValidationError` if argument validation fails.
  - The exception object contains details about the validation errors, including input values and expected types.

Related Concepts:
  - Validators (see Pydantic documentation for details on underlying validation mechanisms).
  - Type Annotations (PEP 526).
  - Data Conversion in Pydantic Models.
  - Strict Mode for Pydantic validation.
```

----------------------------------------

TITLE: Use Stdlib Dataclasses with Pydantic BaseModel
DESCRIPTION: Explains how standard library dataclasses used within Pydantic models undergo validation. Covers the use of `ConfigDict(revalidate_instances='always')` and handling frozen dataclasses.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/dataclasses.md#_snippet_5

LANGUAGE: python
CODE:
```
import dataclasses
from typing import Optional

from pydantic import BaseModel, ConfigDict, ValidationError


@dataclasses.dataclass(frozen=True)
class User:
    name: str


class Foo(BaseModel):
    # Required so that pydantic revalidates the model attributes:
    model_config = ConfigDict(revalidate_instances='always')

    user: Optional[User] = None


# nothing is validated as expected:
user = User(name=['not', 'a', 'string'])
print(user)
#> User(name=['not', 'a', 'string'])


try:
    Foo(user=user)
except ValidationError as e:
    print(e)
    """
    1 validation error for Foo
    user.name
      Input should be a valid string [type=string_type, input_value=['not', 'a', 'string'], input_type=list]
    """

foo = Foo(user=User(name='pika'))
try:
    foo.user.name = 'bulbi'
except dataclasses.FrozenInstanceError as e:
    print(e)
    #> cannot assign to field 'name'

```

----------------------------------------

TITLE: Pydantic Core Schema and Validation API References
DESCRIPTION: This section provides references to key classes and methods within Pydantic and `pydantic-core` that are central to schema generation, validation, and serialization. It outlines their roles in handling core schemas, JSON schemas, and model operations.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/internals/architecture.md#_snippet_8

LANGUAGE: APIDOC
CODE:
```
pydantic.json_schema.GenerateJsonSchema:
  Description: Class responsible for generating JSON Schema from a core schema.
  Methods:
    generate(core_schema: CoreSchema) -> dict: Main entry point for JSON Schema generation.
    bool_schema(bool_core_schema: CoreSchema) -> dict: Generates JSON Schema for boolean types.

pydantic_core.SchemaValidator:
  Description: Class for validating data against a core schema.
  Methods:
    validate_python(data: Any) -> Any: Validates Python data against the model's core schema.

pydantic_core.SchemaSerializer:
  Description: Class for serializing data from a core schema.
  Methods:
    to_python(instance: Any) -> Any: Serializes a model instance's data based on its core schema.

pydantic.GetCoreSchemaHandler:
  Description: Handler passed to __get_pydantic_core_schema__ for recursive schema generation.
  Methods:
    __call__(source: Any) -> CoreSchema: Recursively calls other __get_pydantic_core_schema__ methods.
```

----------------------------------------

TITLE: Pydantic create_model Default Module
DESCRIPTION: Changes the default value of the `__module__` argument in `create_model` from `None` to `'pydantic.main'`, improving pickling support for created models.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_211

LANGUAGE: python
CODE:
```
from pydantic import create_model

# Previously, __module__ defaulted to None, which could cause pickling issues.
# Now, it defaults to 'pydantic.main' for better pickling.

MyDynamicModel = create_model('MyDynamicModel', __module__='my_app.models', field1=(str, ...))

# If __module__ is not provided, it will be 'pydantic.main' by default.
# AnotherDynamicModel = create_model('AnotherDynamicModel', field2=(int, 0))

```

----------------------------------------

TITLE: Pydantic Schema Ref Template
DESCRIPTION: Introduces support for `ref_template` when creating schema `$ref`s, allowing customization of how references are formatted in generated schemas.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_215

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field

class Item(BaseModel):
    name: str

class Container(BaseModel):
    item: Item

    class Config:
        # Custom template for schema references
        # Example: Use a specific prefix for all internal references
        ref_template = '#/components/schemas/{model}'

# When generating schema for Container, the reference to Item will use the custom template.
```

----------------------------------------

TITLE: Pydantic Mypy Plugin Setting: warn_required_dynamic_aliases
DESCRIPTION: Documentation for the `warn_required_dynamic_aliases` setting, explaining its purpose when using dynamically-determined aliases or alias generators on a model with `validate_by_name` set to `False`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/integrations/mypy.md#_snippet_6

LANGUAGE: APIDOC
CODE:
```
warn_required_dynamic_aliases: bool
  Description: Whether to error when using a dynamically-determined alias or alias generator on a model with `validate_by_name` set to `False`.
  Impact: If such aliases are present, mypy cannot properly type check calls to `__init__`. In this case, it will default to treating all arguments as not required.
```

----------------------------------------

TITLE: Pydantic: ConfigDict Extra Data Options
DESCRIPTION: Details the three possible values for the `extra` configuration option in Pydantic's `ConfigDict`: 'ignore' (default), 'forbid', and 'allow'.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_15

LANGUAGE: APIDOC
CODE:
```
PydanticConfigDict.extra:
  Controls how Pydantic handles fields not defined in the model.
  
  Values:
  - 'ignore': (Default) Extra fields are ignored and not stored.
  - 'forbid': Extra fields cause a validation error.
  - 'allow': Extra fields are allowed and stored in the `__pydantic_extra__` attribute.
  
  Example:
  ```python
  from pydantic import BaseModel, ConfigDict

  class MyModel(BaseModel):
      id: int
      model_config = ConfigDict(extra='allow')
  
  # Instance with extra data
  m = MyModel(id=1, name='test', value=100)
  # m.model_dump() -> {'id': 1, 'name': 'test', 'value': 100}
  # m.__pydantic_extra__ -> {'name': 'test', 'value': 100}
  ```
```

----------------------------------------

TITLE: Pydantic V2 Regex Engine and Performance
DESCRIPTION: Explains that Pydantic V2 uses the Rust `regex` crate instead of Python's `re` library for pattern validation. This change offers linear time searching and drops lookarounds/backreferences for performance and security, with an option to revert to Python's regex via `regex_engine` config.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_25

LANGUAGE: APIDOC
CODE:
```
Pydantic Regex Engine:
  - Uses Rust `regex` crate by default.
  - Offers linear time searching.
  - Drops lookarounds and backreferences for performance/security.
  - For Python `re` compatibility: use `ConfigDict(regex_engine='python')`.
```

----------------------------------------

TITLE: TypeAdapter for Parsing Data into BaseModel Lists
DESCRIPTION: Illustrates using TypeAdapter to parse raw data into a list of Pydantic BaseModel instances. This functionality is similar to BaseModel.model_validate but works with any Pydantic-compatible type, including lists of models.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/type_adapter.md#_snippet_1

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, TypeAdapter


class Item(BaseModel):
    id: int
    name: str


# `item_data` could come from an API call, eg., via something like:
# item_data = requests.get('https://my-api.com/items').json()
item_data = [{'id': 1, 'name': 'My Item'}]

items = TypeAdapter(list[Item]).validate_python(item_data)
print(items)
#> [Item(id=1, name='My Item')]
```

----------------------------------------

TITLE: Disabling Pydantic Experimental Warnings
DESCRIPTION: Provides a Python code snippet to suppress `PydanticExperimentalWarning` messages. This is useful for users who want to avoid repeated warnings when working with experimental features.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/version-policy.md#_snippet_1

LANGUAGE: python
CODE:
```
import warnings

from pydantic import PydanticExperimentalWarning

warnings.filterwarnings('ignore', category=PydanticExperimentalWarning)
```

----------------------------------------

TITLE: Pydantic Typing Utilities
DESCRIPTION: A collection of type aliases and utility functions for working with Python's typing system within Pydantic. These include custom type definitions and helper functions for introspection and manipulation of type hints.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_36

LANGUAGE: APIDOC
CODE:
```
pydantic.typing:
  AbstractSetIntStr: Alias for typing.AbstractSet[IntStr]
  AnyCallable: Alias for typing.Callable
  AnyClassMethod: Alias for typing.Callable[..., Any]
  CallableGenerator: Alias for typing.Callable[..., Generator[Any, None, None]]
  DictAny: Alias for typing.Dict[Any, Any]
  DictIntStrAny: Alias for typing.Dict[IntStr, Any]
  DictStrAny: Alias for typing.Dict[str, Any]
  IntStr: Alias for typing.Union[int, str]
  ListStr: Alias for typing.List[str]
  MappingIntStrAny: Alias for typing.Mapping[IntStr, Any]
  NoArgAnyCallable: Alias for typing.Callable[[], Any]
  NoneType: Alias for type(None)
  ReprArgs: Alias for typing.Tuple[str, ...]
  SetStr: Alias for typing.Set[str]
  StrPath: Alias for typing.Union[str, pathlib.Path]
  TupleGenerator: Alias for typing.Tuple[Generator[Any, None, None], ...]
  WithArgsTypes: Alias for typing.Union[typing.Tuple[type, ...], typing.Tuple[typing.TypeVar, ...]]
  all_literal_values(tp: type) -> typing.List[Any]
    Returns a list of all values for a Literal type.
  display_as_type(tp: type) -> str
    Returns a string representation of a type.
  get_all_type_hints(tp: type, include_extras: bool = False) -> typing.Dict[str, type]
    Get all type hints for a class, including inherited ones.
  get_args(tp: type) -> typing.Tuple[type, ...]
    Get the arguments of a generic type.
  get_origin(tp: type) -> typing.Optional[type]
    Get the origin of a generic type.
  get_sub_types(tp: type) -> typing.List[type]
    Get the subtypes of a union type.
  is_callable_type(tp: type) -> bool
    Check if a type is a callable type.
  is_classvar(tp: type) -> bool
    Check if a type is a ClassVar.
  is_finalvar(tp: type) -> bool
    Check if a type is a Final variable.
  is_literal_type(tp: type) -> bool
    Check if a type is a Literal type.
  is_namedtuple(tp: type) -> bool
    Check if a type is a namedtuple.
  is_new_type(tp: type) -> bool
    Check if a type is a NewType.
  is_none_type(tp: type) -> bool
    Check if a type is None.
  is_typeddict(tp: type) -> bool
    Check if a type is a TypedDict.
  is_typeddict_special(tp: type) -> bool
    Check if a type is a special TypedDict.
  is_union(tp: type) -> bool
    Check if a type is a Union.
  new_type_supertype(tp: type) -> type
    Get the supertype of a NewType.
  resolve_annotations(tp: type) -> typing.Dict[str, type]
    Resolve forward references in type annotations.
  typing_base(tp: type) -> type
    Get the base type of a generic type.
  update_field_forward_refs(tp: type, **localns: typing.Any) -> None
    Update forward references in a model's fields.
  update_model_forward_refs(tp: type, **localns: typing.Any) -> None
    Update forward references in a model's fields.
```

----------------------------------------

TITLE: Fix configuration declarations as kwargs
DESCRIPTION: Corrects a bug where configuration declarations passed as keyword arguments during class creation were not handled properly. This ensures configurations are applied as expected.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_129

LANGUAGE: python
CODE:
```
Fix bug with configurations declarations that are passed as
  keyword arguments during class creation, [#2532](https://github.com/pydantic/pydantic/pull/2532) by @uriyyo
```

----------------------------------------

TITLE: Partial JSON Validation with `experimental_allow_partial`
DESCRIPTION: Demonstrates using `TypeAdapter` with `experimental_allow_partial=True` to validate JSON strings where the last field might be incomplete or invalid. Errors in the last field are ignored, allowing partial data to pass validation.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/experimental.md#_snippet_14

LANGUAGE: python
CODE:
```
from typing import Annotated

from annotated_types import MinLen
from typing_extensions import TypedDict

from pydantic import TypeAdapter


class Foobar(TypedDict, total=False):
    a: int
    b: Annotated[str, MinLen(5)]


ta = TypeAdapter(Foobar)

v = ta.validate_json(
    '{"a": 1, "b": "12', experimental_allow_partial=True  # (1)!
)
print(v)
#> {'a': 1}

v = ta.validate_json(
    '{"a": 1, "b": "12"}', experimental_allow_partial=True  # (2)!
)
print(v)
#> {'a': 1}
```

----------------------------------------

TITLE: JSON Schema for String Constraints
DESCRIPTION: Illustrates the JSON schema generated by Pydantic for string constraints, showing how 'min_length', 'max_length', and 'pattern' are translated to 'minLength', 'maxLength', and 'pattern' respectively.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/fields.md#_snippet_21

LANGUAGE: json
CODE:
```
{
  "title": "Foo",
  "type": "object",
  "properties": {
    "short": {
      "title": "Short",
      "type": "string",
      "minLength": 3
    },
    "long": {
      "title": "Long",
      "type": "string",
      "maxLength": 10
    },
    "regex": {
      "title": "Regex",
      "type": "string",
      "pattern": "^\\d*$"
    }
  },
  "required": [
    "short",
    "long",
    "regex"
  ]
}
```

----------------------------------------

TITLE: Pydantic Handling of `typing.Any`
DESCRIPTION: Describes Pydantic's behavior when encountering `typing.Any`, indicating it allows any value, including `None`, without performing specific validation beyond basic type checking.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/standard_library_types.md#_snippet_31

LANGUAGE: APIDOC
CODE:
```
typing.Any:
  Description: Allows any value, including `None`.
```

----------------------------------------

TITLE: Pydantic Annotated Field Constraints
DESCRIPTION: Demonstrates using `Annotated` with Pydantic `Field` for constraints, highlighting potential issues with compound types and recommending `Annotated` for safety.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/fields.md#_snippet_19

LANGUAGE: python
CODE:
```
from typing import Annotated, Optional

from pydantic import BaseModel, Field


class Foo(BaseModel):
    positive: Optional[Annotated[int, Field(gt=0)]]
    # Can error in some cases, not recommended:
    non_negative: Optional[int] = Field(ge=0)

```

----------------------------------------

TITLE: Settings and Aliases
DESCRIPTION: Fixes and improvements related to the use of aliases within Pydantic settings.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_297

LANGUAGE: APIDOC
CODE:
```
Settings Aliases:
  - Fixed alias use in settings.
```

----------------------------------------

TITLE: Pydantic Data Validation Methods
DESCRIPTION: This section documents the primary methods Pydantic offers for validating and parsing data into model instances. It covers their purpose, typical usage, and how they handle validation errors.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_20

LANGUAGE: APIDOC
CODE:
```
Pydantic Data Validation:

Provides methods on model classes for parsing and validating data.

1. model_validate(data: Any, *, strict: bool | None = None, context: dict | None = None)
   - Parses and validates data from a dictionary or object into a Pydantic model instance.
   - Similar to the model's __init__ but accepts a dictionary or object directly.
   - Raises ValidationError if the data cannot be validated or is not a dictionary/model instance.
   - Parameters:
     - data: The dictionary or object to validate.
     - strict: If True, disables all coercion and validation.
     - context: Optional context to pass to validators.
   - Returns: A validated model instance.
   - Example:
     ```python
     from pydantic import BaseModel
     class User(BaseModel):
         id: int
     m = User.model_validate({'id': 123})
     ```

2. model_validate_json(json_data: str | bytes | bytearray, *, strict: bool | None = None, context: dict | None = None)
   - Parses and validates data from a JSON string or bytes object into a Pydantic model instance.
   - Generally faster than manually parsing JSON to a dictionary before validation.
   - Raises ValidationError for invalid JSON or validation failures.
   - Parameters:
     - json_data: The JSON string or bytes to validate.
     - strict: If True, disables all coercion and validation.
     - context: Optional context to pass to validators.
   - Returns: A validated model instance.
   - Example:
     ```python
     from pydantic import BaseModel
     class User(BaseModel):
         id: int
     m = User.model_validate_json('{"id": 123}')
     ```

3. model_validate_strings(data: dict, *, strict: bool | None = None, context: dict | None = None)
   - Parses and validates data from a dictionary with string keys and values.
   - Validates data in JSON mode, allowing strings to be coerced into correct types.
   - Useful for data originating from non-JSON sources where values might be strings.
   - Raises ValidationError for validation failures.
   - Parameters:
     - data: The dictionary with string keys and values to validate.
     - strict: If True, disables all coercion and validation.
     - context: Optional context to pass to validators.
   - Returns: A validated model instance.
   - Example:
     ```python
     from pydantic import BaseModel
     class User(BaseModel):
         id: int
     m = User.model_validate_strings({'id': '123'})
     ```

Related Concepts:
- ValidationError: Raised when data validation fails.
- ConfigDict.revalidate_instances: Controls whether model instances passed to model_validate are re-validated.
```

----------------------------------------

TITLE: Validate INI Data with Pydantic in Python
DESCRIPTION: Validates data from an INI file using Pydantic. It uses Python's `configparser` to read the INI file and then validates a specific section against a Pydantic model. Requires `pydantic` library.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/examples/files.md#_snippet_8

LANGUAGE: python
CODE:
```
import configparser

from pydantic import BaseModel, EmailStr, PositiveInt


class Person(BaseModel):
    name: str
    age: PositiveInt
    email: EmailStr


config = configparser.ConfigParser()
config.read('person.ini')
person = Person.model_validate(config['PERSON'])
print(person)
#> name='John Doe' age=30 email='john@example.com'
```

----------------------------------------

TITLE: Fetch and Validate List of Users with httpx and TypeAdapter
DESCRIPTION: Retrieves a list of users from the JSONPlaceholder API using httpx and validates the entire list using Pydantic's TypeAdapter. This showcases handling collections of Pydantic models efficiently.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/examples/requests.md#_snippet_1

LANGUAGE: python
CODE:
```
from pprint import pprint

import httpx

from pydantic import BaseModel, EmailStr, TypeAdapter


class User(BaseModel):
    id: int
    name: str
    email: EmailStr


url = 'https://jsonplaceholder.typicode.com/users/'  # (1)!

response = httpx.get(url)
response.raise_for_status()

users_list_adapter = TypeAdapter(list[User])

users = users_list_adapter.validate_python(response.json())
pprint([u.name for u in users])
"""
['Leanne Graham',
 'Ervin Howell',
 'Clementine Bauch',
 'Patricia Lebsack',
 'Chelsey Dietrich',
 'Mrs. Dennis Schulist',
 'Kurtis Weissnat',
 'Nicholas Runolfsdottir V',
 'Glenna Reichert',
 'Clementina DuBuque']
"""
```

----------------------------------------

TITLE: Pydantic JSON in Environment Variables
DESCRIPTION: Explains the feature allowing JSON strings in environment variables to be parsed for complex Pydantic types, simplifying configuration management.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_306

LANGUAGE: APIDOC
CODE:
```
Pydantic JSON in Environment Variables:

Allows complex types (like lists or dictionaries) to be configured via environment variables by providing their values as JSON strings.
Example:
  MY_SETTING='{"key": "value"}'
  MY_LIST='[1, 2, 3]'
```

----------------------------------------

TITLE: Update Version and Lock File
DESCRIPTION: Manually edits the `pydantic/version.py` file to set the new version number. Subsequently, it updates the lock file using `uv lock` to reflect these changes.

SOURCE: https://github.com/pydantic/pydantic/blob/main/release/README.md#_snippet_3

LANGUAGE: shell
CODE:
```
uv lock -P pydantic
```

----------------------------------------

TITLE: Partial JSON Dictionary Parsing with pydantic_core.from_json
DESCRIPTION: Illustrates parsing incomplete JSON dictionaries using `pydantic_core.from_json` with `allow_partial=True`. This enables deserialization of JSON objects even when they are missing closing braces or trailing commas.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/json.md#_snippet_2

LANGUAGE: python
CODE:
```
from pydantic_core import from_json

partial_dog_json = '{"breed": "lab", "name": "fluffy", "friends": ["buddy", "spot", "rufus"], "age'

dog_dict = from_json(partial_dog_json, allow_partial=True)
print(dog_dict)
#> {'breed': 'lab', 'name': 'fluffy', 'friends': ['buddy', 'spot', 'rufus']}
```

----------------------------------------

TITLE: JSON Schema Generation Customization with GenerateJsonSchema
DESCRIPTION: Shows how to customize JSON schema generation in Pydantic V2 by subclassing `GenerateJsonSchema` and passing the custom generator to methods like `model_json_schema` or `TypeAdapter.json_schema`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_30

LANGUAGE: python
CODE:
```
from pydantic.json_schema import GenerateJsonSchema

class CustomSchemaGenerator(GenerateJsonSchema):
    # Override methods here to customize schema generation
    pass

# Example usage with BaseModel
# MyModel.model_json_schema(schema_generator=CustomSchemaGenerator)

# Example usage with TypeAdapter
# adapter.json_schema(schema_generator=CustomSchemaGenerator)
```

----------------------------------------

TITLE: Type Support and Additions
DESCRIPTION: Information on new types supported by Pydantic, including custom types and standard library types.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_295

LANGUAGE: APIDOC
CODE:
```
New Types:
  - Added `UrlStr` and `urlstr` types.
  - Added `FilePath` and `DirectoryPath` types.
  - Added `Json` type support.
  - Added `NewType` support.
  - Allowed `Pattern` type.
  - Allowed arbitrary types in models.

Type Handling:
  - Allowed `timedelta` objects as values for properties of type `timedelta`.
  - Supported tuples.
```

----------------------------------------

TITLE: Import Pydantic V1 BaseModel
DESCRIPTION: Shows how to import the BaseModel class from the Pydantic V1 namespace. This is part of the strategy to use Pydantic V1 features within a Pydantic V2 environment.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_4

LANGUAGE: python
CODE:
```
from pydantic.v1 import BaseModel
```

----------------------------------------

TITLE: ConfigDict validate_by_alias and validate_by_name
DESCRIPTION: Demonstrates using both `validate_by_alias=True` and `validate_by_name=True`, allowing validation via either alias or attribute name. Shows validation using `my_alias` and `my_field`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/alias.md#_snippet_8

LANGUAGE: Python
CODE:
```
from pydantic import BaseModel, ConfigDict, Field


class Model(BaseModel):
    my_field: str = Field(validation_alias='my_alias')

    model_config = ConfigDict(validate_by_alias=True, validate_by_name=True)


print(repr(Model(my_alias='foo')))  # (1)!
#> Model(my_field='foo')

print(repr(Model(my_field='foo')))  # (2)!
#> Model(my_field='foo')
```

----------------------------------------

TITLE: Static Model with Aliases, Descriptions, and Private Attributes
DESCRIPTION: Defines a static Pydantic model with field aliases, descriptions via `Annotated`, and private attributes.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_44

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field, PrivateAttr
from typing import Annotated

class StaticModel(BaseModel):
    foo: str = Field(alias='FOO')
    bar: Annotated[str, Field(description='Bar field')]
    _private: int = PrivateAttr(default=1)
```

----------------------------------------

TITLE: Pydantic TypeVar Support
DESCRIPTION: Demonstrates Pydantic's support for Python's TypeVar, including unconstrained, bound, and constrained TypeVars. Shows how these generic types are used within Pydantic models for flexible data validation.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/standard_library_types.md#_snippet_23

LANGUAGE: python
CODE:
```
from typing import TypeVar

from pydantic import BaseModel

Foobar = TypeVar('Foobar')
BoundFloat = TypeVar('BoundFloat', bound=float)
IntStr = TypeVar('IntStr', int, str)


class Model(BaseModel):
    a: Foobar  # equivalent of ": Any"
    b: BoundFloat  # equivalent of ": float"
    c: IntStr  # equivalent of ": Union[int, str]"


print(Model(a=[1], b=4.2, c='x'))
#> a=[1] b=4.2 c='x'

# a may be None
print(Model(a=None, b=1, c=1))
#> a=None b=1.0 c=1
```

----------------------------------------

TITLE: Use Any to Skip Validation
DESCRIPTION: When a value does not require validation and should be kept unchanged, use `typing.Any`. This avoids unnecessary processing and potential errors from strict type checking.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/performance.md#_snippet_1

LANGUAGE: python
CODE:
```
from typing import Any

from pydantic import BaseModel


class ModelWithAny(BaseModel):
    a: Any


model = ModelWithAny(a=1)
# model.a will be 1, no validation performed on 'a'
```

----------------------------------------

TITLE: Pydantic Backward Compatibility: Dataclass and BaseModel Interaction
DESCRIPTION: Illustrates Pydantic's backward compatibility logic for schema generation when a dataclass references a BaseModel that is not yet defined in the current scope. It shows how Pydantic includes the parent namespace and the BaseModel itself in locals to resolve annotations like 'Bar | None'.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/internals/resolving_annotations.md#_snippet_5

LANGUAGE: python
CODE:
```
from dataclasses import dataclass

from pydantic import BaseModel


@dataclass
class Foo:
    a: 'Bar | None' = None


class Bar(BaseModel):
    b: Foo
```

----------------------------------------

TITLE: Apply Pydantic Decorator to Stdlib Dataclass
DESCRIPTION: Illustrates applying the Pydantic dataclass decorator directly to an existing standard library dataclass. This creates a Pydantic-enhanced subclass, enabling validation on instantiation.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/dataclasses.md#_snippet_4

LANGUAGE: python
CODE:
```
import dataclasses

import pydantic


@dataclasses.dataclass
class A:
    a: int

PydanticA = pydantic.dataclasses.dataclass(A)
print(PydanticA(a='1'))
#> A(a=1)

```

----------------------------------------

TITLE: Root Model with Customization
DESCRIPTION: Illustrates how to customize Root Model behavior, such as defining custom serialization or adding extra attributes that are not part of the root value.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/root_model.md#_snippet_2

LANGUAGE: python
CODE:
```
from pydantic import RootModel, Field
from typing import List

class ItemList(RootModel[List[str]]):
    item_count: int = Field(default=0)

    def __init__(self, **data):
        super().__init__(**data)
        self.item_count = len(self.root)

# Example usage:
items = ItemList(["apple", "banana"])
print(items.root)
print(items.item_count)
# Output:
# ['apple', 'banana']
# 2
```

----------------------------------------

TITLE: ConfigDict validate_by_name
DESCRIPTION: Shows how `ConfigDict.validate_by_name=True` allows validation using the attribute name, overriding alias usage. Demonstrates validation with `my_field`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/alias.md#_snippet_7

LANGUAGE: Python
CODE:
```
from pydantic import BaseModel, ConfigDict, Field


class Model(BaseModel):
    my_field: str = Field(validation_alias='my_alias')

    model_config = ConfigDict(validate_by_alias=False, validate_by_name=True)


print(repr(Model(my_field='foo')))  # (1)!
#> Model(my_field='foo')
```

----------------------------------------

TITLE: Enable Mypy Linting in VS Code
DESCRIPTION: Configures VS Code to use mypy for static type checking, which can detect errors missed by Pylance. This integration allows inline display of mypy errors, including those from the Pydantic mypy plugin.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/integrations/visual_studio_code.md#_snippet_1

LANGUAGE: VSCode Settings
CODE:
```
{
  "python.linting.mypyEnabled": true
}
```

----------------------------------------

TITLE: Percent Encoding in AnyUrl
DESCRIPTION: Implements percent encoding for `AnyUrl` and its descendant types, ensuring URLs are correctly formatted and safe for network transmission.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_112

LANGUAGE: python
CODE:
```
from pydantic import AnyUrl

class UrlModel(BaseModel):
    url: AnyUrl

# Example:
# url_with_space = UrlModel(url='http://example.com/path with spaces')
# print(url_with_space.url) # Output: http://example.com/path%20with%20spaces
# This ensures proper URL encoding.
```

----------------------------------------

TITLE: Pydantic Strict Mode vs. Lax Mode Validation
DESCRIPTION: Demonstrates Pydantic's strict mode by comparing validation behavior with and without the `strict=True` parameter. It shows how strict mode prevents type coercion for integer fields, raising a ValidationError for string inputs.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/strict_mode.md#_snippet_0

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ValidationError


class MyModel(BaseModel):
    x: int


print(MyModel.model_validate({'x': '123'}))  # lax mode
#> x=123

try:
    MyModel.model_validate({'x': '123'}, strict=True)  # strict mode
except ValidationError as exc:
    print(exc)
    """
    1 validation error for MyModel
    x
      Input should be a valid integer [type=int_type, input_value='123', input_type=str]
    """

```

----------------------------------------

TITLE: Pydantic Field vs. Type Metadata with Annotated
DESCRIPTION: Highlights the difference in applying metadata to fields versus types using Annotated and Field(), and the impact on features like deprecation.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/fields.md#_snippet_4

LANGUAGE: python
CODE:
```
class Model(BaseModel):
    field_bad: Annotated[int, Field(deprecated=True)] | None = None  # (1)!
    field_ok: Annotated[int | None, Field(deprecated=True)] = None  # (2)!

      # 1. The [`Field()`][pydantic.fields.Field] function is applied to `int` type, hence the
      #    `deprecated` flag won't have any effect. While this may be confusing given that the name of
      #    the [`Field()`][pydantic.fields.Field] function would imply it should apply to the field, 
      #    the API was designed when this function was the only way to provide metadata. You can
      #    alternatively make use of the [`annotated_types`](https://github.com/annotated-types/annotated-types)
      #    library which is now supported by Pydantic.
      #
      # 2. The [`Field()`][pydantic.fields.Field] function is applied to the "top-level" union type, 
      #    hence the `deprecated` flag will be applied to the field.
```

----------------------------------------

TITLE: Pydantic: Root validator `pre=False` and `skip_on_failure`
DESCRIPTION: When using `@root_validator` with `pre=False` (the default), `skip_on_failure` must be set to `True`. The option `skip_on_failure=False` is deprecated and will raise an error. Setting it to `True` prevents the validator from running if any field validation fails.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/usage_errors.md#_snippet_32

LANGUAGE: python
CODE:
```
# Example illustrating the rule, not a direct error-raising snippet
# from pydantic import BaseModel, root_validator
#
# class Model(BaseModel):
#     a: int
#     b: str
#
#     @root_validator(pre=False, skip_on_failure=True)
#     def check_model(cls, values):
#         # ... validation logic ...
#         return values
```

----------------------------------------

TITLE: Pydantic Model Validation
DESCRIPTION: Function for validating Pydantic models.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_38

LANGUAGE: APIDOC
CODE:
```
pydantic.validate_model(model: type, **kwargs: typing.Any) -> typing.Any
  Validate a model with given keyword arguments.
```

----------------------------------------

TITLE: Use TypedDict Over Nested Models for Performance
DESCRIPTION: For nested data structures, using `typing_extensions.TypedDict` can offer significant performance improvements compared to deeply nested Pydantic `BaseModel` classes. Benchmarks show `TypedDict` can be approximately 2.5x faster.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/performance.md#_snippet_4

LANGUAGE: python
CODE:
```
from timeit import timeit

from typing_extensions import TypedDict

from pydantic import BaseModel, TypeAdapter


# Define nested structure using TypedDict
class TypedDictA(TypedDict):
    a: str
    b: int


class TypedDictModel(TypedDict):
    a: TypedDictA


# Define equivalent structure using Pydantic BaseModel
class BaseModelB(BaseModel):
    a: str
    b: int


class BaseModelNested(BaseModel):
    b: BaseModelB


# Performance comparison using timeit
ta = TypeAdapter(TypedDictModel)

# Validate data using TypedDict
result_typeddict = timeit(
    lambda: ta.validate_python({'a': {'a': 'a', 'b': 2}}), number=10000
)

# Validate data using Pydantic BaseModel
result_basemodel = timeit(
    lambda: BaseModelNested.model_validate({'b': {'a': 'a', 'b': 2}}), number=10000
)

# Print the performance ratio (BaseModel vs TypedDict)
print(f"Performance ratio (BaseModel/TypedDict): {result_basemodel / result_typeddict:.2f}")
```

----------------------------------------

TITLE: Performance Improvements
DESCRIPTION: Specific optimizations implemented to enhance build time performance, particularly in the creation of Pydantic models and core schema generation.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_15

LANGUAGE: python
CODE:
```
Create a single dictionary when creating a `CoreConfig` instance
```

----------------------------------------

TITLE: Pydantic Handling of `typing.Pattern`
DESCRIPTION: Explains how Pydantic processes `typing.Pattern` types, indicating that the input value will be passed to `re.compile(v)` to create a regular expression pattern for validation.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/standard_library_types.md#_snippet_34

LANGUAGE: APIDOC
CODE:
```
typing.Pattern:
  Behavior: Will cause the input value to be passed to `re.compile(v)` to create a regular expression pattern.
```

----------------------------------------

TITLE: Adding a New Test Case to Mypy Configuration
DESCRIPTION: Demonstrates how to add a new Python test file to the Mypy test suite. This involves placing the file in the modules directory and referencing it within the `cases` list in the main test file, alongside its corresponding Mypy configuration.

SOURCE: https://github.com/pydantic/pydantic/blob/main/tests/mypy/README.md#_snippet_2

LANGUAGE: python
CODE:
```
# modules/new_test.py

from pydantic import BaseModel

class Model(BaseModel):
    a: int


model = Model(a=1, b=2)
```

LANGUAGE: python
CODE:
```
cases: list[ParameterSet | tuple[str, str]] = [
    ...
    # One-off cases
    *[
            ('mypy-plugin.ini', 'custom_constructor.py'),
            ('mypy-plugin.ini', 'config_conditional_extra.py'),
            ...
            ('mypy-plugin.ini', 'new_test.py'),  # <-- new test added.
        ]
    ]
```

----------------------------------------

TITLE: Basic Static Model Definition
DESCRIPTION: Defines a static Pydantic model with simple field types and default values.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_42

LANGUAGE: python
CODE:
```
class StaticFoobarModel(BaseModel):
    foo: str
    bar: int = 123
```

----------------------------------------

TITLE: Pydantic Generics with TypeVar Bounds
DESCRIPTION: Illustrates the use of `typing.TypeVar` with bounds and without bounds in generic Pydantic models. Shows how generic types can be inferred or explicitly parameterized, and their serialization behavior.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_39

LANGUAGE: python
CODE:
```
from typing import Generic, TypeVar

from pydantic import BaseModel

TBound = TypeVar('TBound', bound=BaseModel)
TNoBound = TypeVar('TNoBound')


class IntValue(BaseModel):
    value: int


class ItemBound(BaseModel, Generic[TBound]):
    item: TBound


class ItemNoBound(BaseModel, Generic[TNoBound]):
    item: TNoBound


item_bound_inferred = ItemBound(item=IntValue(value=3))
item_bound_explicit = ItemBound[IntValue](item=IntValue(value=3))
item_no_bound_inferred = ItemNoBound(item=IntValue(value=3))
item_no_bound_explicit = ItemNoBound[IntValue](item=IntValue(value=3))

# calling `print(x.model_dump())` on any of the above instances results in the following:
#> {'item': {'value': 3}}
```

----------------------------------------

TITLE: RootModel with Custom Iteration and Item Access
DESCRIPTION: Shows how to implement `__iter__` and `__getitem__` on a `RootModel` subclass to allow direct iteration and item access.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_48

LANGUAGE: python
CODE:
```
from pydantic import RootModel


class Pets(RootModel):
    root: list[str]

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]


pets = Pets.model_validate(['dog', 'cat'])
print(pets[0])
#> dog
print([pet for pet in pets])
#> ['dog', 'cat']
```

----------------------------------------

TITLE: create_model with __config__
DESCRIPTION: Fixed the `create_model` function to correctly use the passed `__config__` attribute, ensuring proper model configuration when dynamically creating models.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_292

LANGUAGE: python
CODE:
```
from pydantic import create_model, BaseModel

class MyConfig:
    extra = 'allow'

DynamicModel = create_model('DynamicModel', __config__=MyConfig, field1=(str, ...))
```

----------------------------------------

TITLE: Ad-hoc Parsing Functions
DESCRIPTION: Introduces `parse_obj_as` and `parse_file_as` functions for ad-hoc parsing of data into arbitrary pydantic-compatible types. These functions are useful for validating and transforming data without needing to define a full Pydantic model.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_249

LANGUAGE: APIDOC
CODE:
```
parse_obj_as(type_, obj)
parse_file_as(type_, file_path, **kwargs)

# Example:
from typing import List
from pydantic import parse_obj_as

data = [1, 2, 3]
numbers = parse_obj_as(List[int], data)
print(numbers)
# Output: [1, 2, 3]

```

----------------------------------------

TITLE: Pydantic Annotated Pattern for Type Constraints
DESCRIPTION: Demonstrates applying validation constraints to elements within a list using the Annotated pattern with Field().

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/fields.md#_snippet_3

LANGUAGE: python
CODE:
```
from typing import Annotated

from pydantic import BaseModel, Field


class Model(BaseModel):
    int_list: list[Annotated[int, Field(gt=0)]]
    # Valid: [1, 3]
    # Invalid: [-1, 2]
```

----------------------------------------

TITLE: Pydantic Validator Ordering with Annotated Pattern
DESCRIPTION: Demonstrates the execution order of validators when using the annotated pattern in Pydantic. 'WrapValidator' and 'BeforeValidator' run right-to-left, while 'AfterValidator' runs left-to-right.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/validators.md#_snippet_19

LANGUAGE: python
CODE:
```
from typing import Annotated
from pydantic import AfterValidator, BaseModel, BeforeValidator, WrapValidator

def runs_1st(): pass
def runs_2nd(): pass
def runs_3rd(): pass
def runs_4th(): pass

class Model(BaseModel):
    name: Annotated[
        str,
        AfterValidator(runs_3rd),
        AfterValidator(runs_4th),
        BeforeValidator(runs_2nd),
        WrapValidator(runs_1st),
    ]
```

----------------------------------------

TITLE: Pydantic Strict Mode Configuration and Usage
DESCRIPTION: Provides a comprehensive overview of Pydantic's strict mode, detailing its purpose in preventing data coercion and outlining various methods for its implementation. This includes using `strict=True` in validation methods, `Field(strict=True)`, `pydantic.types.Strict` annotations, and `ConfigDict(strict=True)`. It also discusses type coercion nuances between JSON and Python inputs in strict mode.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/strict_mode.md#_snippet_2

LANGUAGE: APIDOC
CODE:
```
Pydantic Strict Mode

Purpose:
By default, Pydantic attempts to coerce values to the desired type (e.g., string '123' to int 123). Strict mode disables or reduces this coercion, causing validation to error if data is not of the exact correct type.

Enabling Strict Mode:
Strict mode can be enabled on a per-model, per-field, or per-validation-call basis.

Methods to Enable Strict Mode:
1.  **Passing `strict=True` to validation methods**: Apply to methods like `BaseModel.model_validate`, `TypeAdapter.validate_python`, and similar JSON validation methods.
2.  **Using `Field(strict=True)`**: Set `strict=True` when defining fields within `BaseModel`, `dataclass`, or `TypedDict`.
3.  **Using `pydantic.types.Strict` as a type annotation**: Annotate fields with `Strict` or Pydantic's provided strict type aliases like `pydantic.types.StrictInt`.
4.  **Using `ConfigDict(strict=True)`**: Set `strict=True` within the `ConfigDict` for a model to apply strict mode globally to that model.

Type Coercions in Strict Mode:
-   **Python Input**: For most types, only instances of the exact type are accepted. Passing floats or strings to an int field will raise `ValidationError`.
-   **JSON Input**: Pydantic is looser when validating from JSON in strict mode. For example, `UUID` fields accept string inputs from JSON, but not from Python.

Refer to the [Conversion Table](conversion_table.md) for detailed type allowances in strict mode.
```

----------------------------------------

TITLE: Pydantic: Validate Password Match with ValidationInfo.data
DESCRIPTION: Demonstrates how to use `ValidationInfo.data` within a Pydantic field validator to access previously validated data, specifically for comparing password fields. It requires `pydantic` and `ValidationInfo` and shows how to raise a `ValueError` if passwords do not match.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/validators.md#_snippet_16

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ValidationInfo, field_validator


class UserModel(BaseModel):
    password: str
    password_repeat: str
    username: str

    @field_validator('password_repeat', mode='after')
    @classmethod
    def check_passwords_match(cls, value: str, info: ValidationInfo) -> str:
        if value != info.data['password']:
            raise ValueError('Passwords do not match')
        return value
```

----------------------------------------

TITLE: TypeAdapter with Union and JSON Schema
DESCRIPTION: Demonstrates how to use Pydantic's TypeAdapter to handle union types (e.g., Union[Cat, Dog]) and generate a corresponding JSON schema. This is useful for defining schemas that can accept multiple distinct data structures.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/json_schema.md#_snippet_3

LANGUAGE: python
CODE:
```
import json
from typing import Union

from pydantic import BaseModel, TypeAdapter


class Cat(BaseModel):
    name: str
    color: str


class Dog(BaseModel):
    name: str
    breed: str


ta = TypeAdapter(Union[Cat, Dog])
ta_schema = ta.json_schema()
print(json.dumps(ta_schema, indent=2))
```

----------------------------------------

TITLE: Pydantic Data Loading Deprecations and Replacements
DESCRIPTION: Details the deprecation of `parse_raw` and `parse_file` in Pydantic V2. `model_validate_json` is the direct replacement for `parse_raw`, while other data loading should involve loading data first and then using `model_validate`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_9

LANGUAGE: APIDOC
CODE:
```
Pydantic Data Loading API Changes:

- `parse_raw` (Deprecated in V2)
  - Purpose: Load data from a JSON string.
  - V2 Replacement: `model_validate_json(json_string)`

- `parse_file` (Deprecated in V2)
  - Purpose: Load data from a file.
  - V2 Approach: Load data from the file (e.g., using `json.load`), then pass to `model_validate`.

- `from_orm` (Deprecated in V2)
  - Purpose: Load data from ORM objects or arbitrary attributes.
  - V2 Replacement: `model_validate(obj, from_attributes=True)`

Example Usage (V2):
```python
from pydantic import BaseModel

class MyModel(BaseModel):
    name: str

# Equivalent to V1's parse_raw
model_instance = MyModel.model_validate_json('{"name": "Alice"}')

# Equivalent to V1's parse_obj with from_attributes=True
class MyORMObject:
    name: str = 'Bob'

model_instance_from_attributes = MyModel.model_validate(MyORMObject(), from_attributes=True)
```
```

----------------------------------------

TITLE: Pydantic v1.0 Fixes and Improvements
DESCRIPTION: Details bug fixes and improvements made in Pydantic v1.0. This includes addressing issues with default values, settings inheritance, string representations, recursive merging for `BaseSettings`, and schema generation for constrained types.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_258

LANGUAGE: APIDOC
CODE:
```
Pydantic Fixes and Improvements (v1.0):

- Fix field of a type with a default value.
- Use `FutureWarning` instead of `DeprecationWarning` for `alias` vs `env` in settings models.
- Fix issue with `BaseSettings` inheritance and `alias` being set to `None`.
- Fix `ConstrainedList` and update schema generation for `min_items`/`max_items` `Field()` arguments.
- Allow abstract sets in `dict()`'s `include`/`exclude` arguments.
- Fix JSON serialization errors on `ValidationError.json()` using `pydantic_encoder`.
- Clarify usage of `remove_untouched` and improve error messages for types with no validators.
- Improve `str`/`repr` logic for `ModelField`.
```

----------------------------------------

TITLE: Validate Incomplete JSON List with Partial Validation
DESCRIPTION: Demonstrates using TypeAdapter.validate_json with experimental_allow_partial=True to validate a JSON string representing an incomplete list of TypedDict objects. Shows how partial data is handled when fields are missing or truncated.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/experimental.md#_snippet_5

LANGUAGE: python
CODE:
```
from typing import Annotated

from annotated_types import MinLen
from typing_extensions import NotRequired, TypedDict

from pydantic import TypeAdapter


class Foobar(TypedDict):
    a: int
    b: NotRequired[float]
    c: NotRequired[Annotated[str, MinLen(5)]]


ta = TypeAdapter(list[Foobar])

v = ta.validate_json('[{"a": 1, "b"', experimental_allow_partial=True)
print(v)
#> [{'a': 1}]
```

----------------------------------------

TITLE: Demonstrating Pydantic MISSING Sentinel Usage
DESCRIPTION: This snippet illustrates the use of Pydantic's `MISSING` sentinel as a default value for a field. It shows how fields with `MISSING` are excluded from serialization and how to check for the sentinel value during runtime. The feature is experimental and relies on PEP 661.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/experimental.md#_snippet_18

LANGUAGE: python
CODE:
```
from typing import Union

from pydantic import BaseModel
from pydantic.experimental.missing_sentinel import MISSING


class Configuration(BaseModel):
    timeout: Union[int, None, MISSING] = MISSING


# configuration defaults, stored somewhere else:
defaults = {'timeout': 200}

conf = Configuration()

# `timeout` is excluded from the serialization output:
print(conf.model_dump())
# Expected output: {}

# The `MISSING` value doesn't appear in the JSON Schema:
# print(Configuration.model_json_schema()['properties']['timeout'])
# Expected output: {'anyOf': [{'type': 'integer'}, {'type': 'null'}], 'title': 'Timeout'}


# `is` can be used to discriminate between the sentinel and other values:
timeout_value = conf.timeout if conf.timeout is not MISSING else defaults['timeout']
print(f"Resolved timeout: {timeout_value}")
# Expected output: Resolved timeout: 200

```

----------------------------------------

TITLE: Advanced Include/Exclude with Sequences in Pydantic
DESCRIPTION: Illustrates advanced usage of `exclude` and `include` with sequences (lists) in Pydantic models. It shows how to exclude specific items within a list or include specific fields of items in a list.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/serialization.md#_snippet_22

LANGUAGE: python
CODE:
```
from pydantic import BaseModel


class Hobby(BaseModel):
    name: str
    info: str


class User(BaseModel):
    hobbies: list[Hobby]


user = User(
    hobbies=[
        Hobby(name='Programming', info='Writing code and stuff'),
        Hobby(name='Gaming', info='Hell Yeah!!!'),
    ],
)

print(user.model_dump(exclude={'hobbies': {-1: {'info'}}}))  # (1)!
"""
{
    'hobbies': [
        {'name': 'Programming', 'info': 'Writing code and stuff'},
        {'name': 'Gaming'},
    ]
}
"""

user.model_dump(
   include={'hobbies': {0: True, -1: {'name'}}}
)

```

----------------------------------------

TITLE: Pydantic BeforeValidator with Decorator
DESCRIPTION: Shows an alternative way to use `BeforeValidator` by applying it as a decorator to a class method. This achieves the same preprocessing as the `Annotated` pattern for ensuring a value is a list before Pydantic's validation.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/validators.md#_snippet_5

LANGUAGE: python
CODE:
```
from typing import Any

from pydantic import BaseModel, ValidationError, field_validator


class Model(BaseModel):
    numbers: list[int]

    @field_validator('numbers', mode='before')
    @classmethod
    def ensure_list(cls, value: Any) -> Any:  # (1)!
        if not isinstance(value, list):  # (2)!
            return [value]
        else:
            return value


print(Model(numbers=2))
#> numbers=[2]
try:
    Model(numbers='str')
except ValidationError as err:
    print(err)  # (3)!
    """
    1 validation error for Model
    numbers.0
      Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='str', input_type=str]
    """

```

----------------------------------------

TITLE: Dataclass Init and InitVar Mutual Exclusion in Pydantic
DESCRIPTION: The `init=False` and `init_var=True` settings for dataclass fields are mutually exclusive. Using both simultaneously results in a `PydanticUserError`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/usage_errors.md#_snippet_47

LANGUAGE: python
CODE:
```
from pydantic import Field
from pydantic.dataclasses import dataclass


@dataclass
class Foo:
    bar: str = Field(init=False, init_var=True)
```

----------------------------------------

TITLE: Support for PEP 695 Generics Syntax
DESCRIPTION: Implements full support for the generics syntax introduced in PEP 695. This allows for more modern and cleaner definition of generic types in Python.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_24

LANGUAGE: python
CODE:
```
# PEP 695 syntax example:
def process_items[T](items: list[T]) -> list[T]:
    return items

# Pydantic models can now leverage this syntax directly.
```

----------------------------------------

TITLE: Configure Pylance Type Checking Mode
DESCRIPTION: Enables Pydantic type error checks within VS Code using the Pylance extension. Setting the 'Type Checking Mode' to 'basic' or 'strict' provides inline error feedback for type mismatches and missing arguments.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/integrations/visual_studio_code.md#_snippet_0

LANGUAGE: VSCode Settings
CODE:
```
{
  "python.analysis.typeCheckingMode": "basic" 
}

// or

{
  "python.analysis.typeCheckingMode": "strict"
}
```

----------------------------------------

TITLE: Pydantic NamedTuple and TypedDict Support
DESCRIPTION: Introduces comprehensive support for `NamedTuple` and `TypedDict` types within `BaseModel` and Pydantic `dataclass`. Utility functions `create_model_from_namedtuple` and `create_model_from_typeddict` are also provided for easier integration.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_152

LANGUAGE: APIDOC
CODE:
```
Pydantic NamedTuple and TypedDict Support:

- Handles and validates `NamedTuple` and `TypedDict` types within `BaseModel` or Pydantic `dataclass`.
- Provides utility functions: `create_model_from_namedtuple` and `create_model_from_typeddict`.
```

----------------------------------------

TITLE: Pydantic: Hypothesis Plugin for Property-Based Testing
DESCRIPTION: Introduces a Hypothesis plugin for Pydantic, simplifying property-based testing with Pydantic's custom types. Detailed usage is available in the Pydantic documentation.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_175

LANGUAGE: python
CODE:
```
# Installation:
# pip install pydantic[hypothesis]

from pydantic import BaseModel
from hypothesis import given
from pydantic_hypothesis import from_type

class MyModel(BaseModel):
    name: str
    age: int

@given(from_type(MyModel))
def test_my_model(model):
    assert isinstance(model, MyModel)
    assert model.age >= 0 # Example assertion

# To run this test, you would typically use pytest or another test runner.
```

----------------------------------------

TITLE: Discriminator with Discriminator Instance
DESCRIPTION: Illustrates using the `discriminator` parameter with a `Discriminator` instance for more complex discrimination logic in Pydantic unions. This allows custom functions to determine the correct model based on input data.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/fields.md#_snippet_26

LANGUAGE: python
CODE:
```
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Discriminator, Field, Tag


class Cat(BaseModel):
    pet_type: Literal['cat']
    age: int


class Dog(BaseModel):
    pet_kind: Literal['dog']
    age: int

def pet_discriminator(v):
    if isinstance(v, dict):
        return v.get('pet_type', v.get('pet_kind'))
    return getattr(v, 'pet_type', getattr(v, 'pet_kind', None))


class Model(BaseModel):
    pet: Union[Annotated[Cat, Tag('cat')], Annotated[Dog, Tag('dog')]] = Field(
        discriminator=Discriminator(pet_discriminator)
    )


print(repr(Model.model_validate({'pet': {'pet_type': 'cat', 'age': 12}})))
#> Model(pet=Cat(pet_type='cat', age=12))

print(repr(Model.model_validate({'pet': {'pet_kind': 'dog', 'age': 12}})))
#> Model(pet=Dog(pet_kind='dog', age=12))
```

----------------------------------------

TITLE: AliasGenerator with Validation and Serialization Aliases
DESCRIPTION: Demonstrates using `AliasGenerator` to define separate uppercase aliases for validation and title-case aliases for serialization within a Pydantic model. Shows how to validate data with aliases and dump the model using aliases.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/alias.md#_snippet_4

LANGUAGE: Python
CODE:
```
from pydantic import AliasGenerator, BaseModel, ConfigDict


class Tree(BaseModel):
    model_config = ConfigDict(
        alias_generator=AliasGenerator(
            validation_alias=lambda field_name: field_name.upper(),
            serialization_alias=lambda field_name: field_name.title(),
        )
    )

    age: int
    height: float
    kind: str


t = Tree.model_validate({'AGE': 12, 'HEIGHT': 1.2, 'KIND': 'oak'})
print(t.model_dump(by_alias=True))
#> {'Age': 12, 'Height': 1.2, 'Kind': 'oak'}
```

----------------------------------------

TITLE: Pydantic BaseModel API Reference
DESCRIPTION: The core class for defining Pydantic data models. Models are created by inheriting from BaseModel and defining fields using annotated attributes. This reference lists key members available on BaseModel instances.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/base_model.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
pydantic.BaseModel
  options:
    show_root_heading: true
    merge_init_into_class: false
    group_by_category: false
    members:
      - __init__
      - model_config
      - model_fields
      - model_computed_fields
      - __pydantic_core_schema__
      - model_extra
      - model_fields_set
      - model_construct
      - model_copy
      - model_dump
      - model_dump_json
      - model_json_schema
      - model_parametrized_name
      - model_post_init
      - model_rebuild
      - model_validate
      - model_validate_json
      - model_validate_strings
```

----------------------------------------

TITLE: Pydantic JSON Schema Type Mapping Priority
DESCRIPTION: Details the priority order Pydantic uses when mapping Python types, custom field types, and constraints to JSON Schema Core, JSON Schema Validation, and OpenAPI Data Types. It also mentions the use of `format` for Pydantic extensions.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/json_schema.md#_snippet_27

LANGUAGE: APIDOC
CODE:
```
Types, custom field types, and constraints (like `max_length`) are mapped to the corresponding spec formats in the following priority order (when there is an equivalent available):

1. JSON Schema Core (https://json-schema.org/draft/2020-12/json-schema-core)
2. JSON Schema Validation (https://json-schema.org/draft/2020-12/json-schema-validation)
3. OpenAPI Data Types (https://github.com/OAI/OpenAPI-Specification/blob/master/versions/3.0.2.md#data-types)
4. The standard `format` JSON field is used to define Pydantic extensions for more complex `string` sub-types.

The field schema mapping from Python or Pydantic to JSON schema is done as follows:

{{ schema_mappings_table }}
```

----------------------------------------

TITLE: Pydantic typing-extensions Dependency
DESCRIPTION: Makes the `typing-extensions` package a required dependency for Pydantic. This ensures access to necessary backported typing features for broader Python version compatibility.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_162

LANGUAGE: APIDOC
CODE:
```
Pydantic typing-extensions Dependency:

`typing-extensions` is now a required dependency.
```

----------------------------------------

TITLE: Model with Unresolved Forward Reference
DESCRIPTION: Demonstrates a Pydantic model with a forward reference ('MyType') that has not yet been defined. Inspecting the `__pydantic_core_schema__` at this stage shows a mock object, indicating the schema generation failed due to the unresolved annotation.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/internals/resolving_annotations.md#_snippet_7

LANGUAGE: python
CODE:
```
from pydantic import BaseModel


class Foo(BaseModel):
    f: 'MyType'


Foo.__pydantic_core_schema__
```

----------------------------------------

TITLE: Pydantic JSON Serialization Methods
DESCRIPTION: Provides access to methods for serializing Pydantic models and data structures into JSON format. These include methods on BaseModel, TypeAdapter, and the underlying pydantic_core library.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/json.md#_snippet_5

LANGUAGE: APIDOC
CODE:
```
pydantic.main.BaseModel.model_dump_json
  Serializes the model instance to a JSON string.
  Parameters:
    - include: Fields to include.
    - exclude: Fields to exclude.
    - by_alias: Use alias names instead of field names.
    - exclude_unset: Exclude fields that were not set.
    - exclude_defaults: Exclude fields that have their default value.
    - exclude_none: Exclude fields that are None.
    - encoder: Custom JSON encoder.
    - mode: Serialization mode ('json' or 'python').
    - indent: Number of spaces for indentation.
    - separators: Tuple of separators.
    - sort_keys: Whether to sort keys.
  Returns: JSON string.

pydantic.type_adapter.TypeAdapter.dump_json
  Serializes data using the type adapter's configuration to a JSON string.
  Parameters:
    - value: The data to serialize.
    - include: Fields to include.
    - exclude: Fields to exclude.
    - by_alias: Use alias names instead of field names.
    - exclude_unset: Exclude fields that were not set.
    - exclude_defaults: Exclude fields that have their default value.
    - exclude_none: Exclude fields that are None.
    - encoder: Custom JSON encoder.
    - mode: Serialization mode ('json' or 'python').
    - indent: Number of spaces for indentation.
    - separators: Tuple of separators.
    - sort_keys: Whether to sort keys.
  Returns: JSON string.

pydantic_core.to_json
  Serializes Python data structures to JSON using pydantic-core.
  Parameters:
    - value: The data to serialize.
    - mode: Serialization mode ('json' or 'python').
    - encoder: Custom JSON encoder.
    - indent: Number of spaces for indentation.
    - separators: Tuple of separators.
    - sort_keys: Whether to sort keys.
  Returns: JSON string.
```

----------------------------------------

TITLE: Schema Generation Logic Moved to GenerateSchema Class
DESCRIPTION: Core schema generation logic for path types and `deque` has been moved into the `GenerateSchema` class. This consolidates schema generation logic and removes workarounds.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_28

LANGUAGE: python
CODE:
```
from pydantic_core import Schema

# Internal refactoring: Schema generation for specific types is now centralized.
# This impacts the internal implementation of schema building.
```

----------------------------------------

TITLE: Pydantic: Correct validator usage with fields
DESCRIPTION: Demonstrates the correct way to define a validator using `@field_validator` by explicitly passing the field name(s) as arguments. This ensures the validator is correctly applied to the intended model fields.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/usage_errors.md#_snippet_27

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, field_validator


class Model(BaseModel):
    a: str

    @field_validator('a')
    def checker(cls, v):
        return v
```

----------------------------------------

TITLE: TypeAdapter Strict Validation for Boolean
DESCRIPTION: Demonstrates using TypeAdapter with strict=True for boolean validation. It shows how strict mode rejects non-strict boolean inputs like the string 'yes', raising a ValidationError.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/strict_mode.md#_snippet_3

LANGUAGE: python
CODE:
```
from pydantic import TypeAdapter, ValidationError

print(TypeAdapter(bool).validate_python('yes'))  # OK: lax
#> True

try:
    TypeAdapter(bool).validate_python('yes', strict=True)  # Not OK: strict
except ValidationError as exc:
    print(exc)
    """
    1 validation error for bool
      Input should be a valid boolean [type=bool_type, input_value='yes', input_type=str]
    """

```

----------------------------------------

TITLE: Pydantic Field Constraints and Schema Generation
DESCRIPTION: Illustrates how Pydantic handles field constraints, including cases where constraints might not be enforced by default and how to explicitly include them in the JSON schema using Field arguments. It shows how to use Field with raw schema attributes for unenforced constraints.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/json_schema.md#_snippet_6

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field, PositiveInt

try:
    # this won't work since `PositiveInt` takes precedence over the
    # constraints defined in `Field`, meaning they're ignored
    class Model(BaseModel):
        foo: PositiveInt = Field(lt=10)

except ValueError as e:
    print(e)


# if you find yourself needing this, an alternative is to declare
# the constraints in `Field` (or you could use `conint()`)
# here both constraints will be enforced:
class ModelB(BaseModel):
    # Here both constraints will be applied and the schema
    # will be generated correctly
    foo: int = Field(gt=0, lt=10)


print(ModelB.model_json_schema())
```

LANGUAGE: json
CODE:
```
{
    "properties": {
        "foo": {
            "exclusiveMaximum": 10,
            "exclusiveMinimum": 0,
            "title": "Foo",
            "type": "integer"
        }
    },
    "required": ["foo"],
    "title": "ModelB",
    "type": "object"
}
```

----------------------------------------

TITLE: Validate and Serialize Data with Redis Queue (Python)
DESCRIPTION: This snippet shows how to use Pydantic models to serialize Python objects into JSON strings for storage in a Redis list (acting as a queue). It also demonstrates deserializing and validating incoming JSON data from Redis back into Pydantic models.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/examples/queues.md#_snippet_0

LANGUAGE: python
CODE:
```
import redis

from pydantic import BaseModel, EmailStr


class User(BaseModel):
    id: int
    name: str
    email: EmailStr


r = redis.Redis(host='localhost', port=6379, db=0)
QUEUE_NAME = 'user_queue'


def push_to_queue(user_data: User) -> None:
    serialized_data = user_data.model_dump_json()
    r.rpush(QUEUE_NAME, serialized_data)
    print(f'Added to queue: {serialized_data}')


user1 = User(id=1, name='John Doe', email='john@example.com')
user2 = User(id=2, name='Jane Doe', email='jane@example.com')

push_to_queue(user1)
# > Added to queue: {"id":1,"name":"John Doe","email":"john@example.com"}

push_to_queue(user2)
# > Added to queue: {"id":2,"name":"Jane Doe","email":"jane@example.com"}


def pop_from_queue() -> None:
    data = r.lpop(QUEUE_NAME)

    if data:
        user = User.model_validate_json(data)
        print(f'Validated user: {repr(user)}')
    else:
        print('Queue is empty')


pop_from_queue()
# > Validated user: User(id=1, name='John Doe', email='john@example.com')

pop_from_queue()
# > Validated user: User(id=2, name='Jane Doe', email='jane@example.com')

pop_from_queue()
# > Queue is empty
```

----------------------------------------

TITLE: Pydantic Optional[conset/conlist] Handling
DESCRIPTION: Fixes an issue where `None` was not correctly allowed for types like `Optional[conset]` or `Optional[conlist]`. This ensures proper validation for optional constrained sets and lists.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_146

LANGUAGE: APIDOC
CODE:
```
Pydantic Optional[conset/conlist] Handling:

Allows `None` for types `Optional[conset]` and `Optional[conlist]`.
```

----------------------------------------

TITLE: Pydantic v2.11.0b1 Packaging Updates
DESCRIPTION: Details packaging changes in Pydantic v2.11.0b1, including the addition of a `check_pydantic_core_version()` function, removal of the `greenlet` development dependency, adoption of the `typing-inspection` library, and bumping `pydantic-core` to v2.31.1.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_13

LANGUAGE: python
CODE:
```
# Packaging Updates in v2.11.0b1
# Add a check_pydantic_core_version() function
# Remove greenlet development dependency
# Use the typing-inspection library
# Bump pydantic-core to v2.31.1
```

----------------------------------------

TITLE: Generate Model Signature with Pydantic
DESCRIPTION: Demonstrates how Pydantic automatically generates a model's signature based on its fields, respecting aliases and default values. This signature is crucial for introspection and integration with libraries like FastAPI.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_56

LANGUAGE: python
CODE:
```
import inspect

from pydantic import BaseModel, Field


class FooModel(BaseModel):
    id: int
    name: str = None
    description: str = 'Foo'
    apple: int = Field(alias='pear')


print(inspect.signature(FooModel))
#> (*, id: int, name: str = None, description: str = 'Foo', pear: int) -> None
```

----------------------------------------

TITLE: Custom Root Models with Mapping Types
DESCRIPTION: Adds support for mapping types when defining custom root models. This allows custom root models to be constructed from or validated against dictionary-like structures more flexibly.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_254

LANGUAGE: APIDOC
CODE:
```
from typing import Dict, List
from pydantic import BaseModel

class MyRootModel(BaseModel):
    __root__: Dict[str, int]


# Example usage:
data = {'a': 1, 'b': 2}
root_model = MyRootModel(__root__=data)
print(root_model.__root__)
# Output: {'a': 1, 'b': 2}

```

----------------------------------------

TITLE: Pydantic Models with Abstract Base Classes
DESCRIPTION: Demonstrates how Pydantic models can inherit from Python's `abc.ABC` to function as abstract base classes. This allows defining abstract methods that must be implemented by concrete subclasses.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_51

LANGUAGE: python
CODE:
```
import abc

from pydantic import BaseModel


class FooBarModel(BaseModel, abc.ABC):
    a: str
    b: int

    @abc.abstractmethod
    def my_abstract_method(self):
        pass
```

----------------------------------------

TITLE: Input Type Preservation for BaseModel and Dataclasses
DESCRIPTION: Shows that Pydantic V2 preserves input types for subclasses of BaseModel and for dataclasses, unlike generic collections where input types are not guaranteed.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_22

LANGUAGE: python
CODE:
```
import pydantic.dataclasses
from pydantic import BaseModel


class InnerModel(BaseModel):
    x: int


class OuterModel(BaseModel):
    inner: InnerModel


class SubInnerModel(InnerModel):
    y: int


m = OuterModel(inner=SubInnerModel(x=1, y=2))
print(m)
#> inner=SubInnerModel(x=1, y=2)


@pydantic.dataclasses.dataclass
class InnerDataclass:
    x: int


@pydantic.dataclasses.dataclass
class SubInnerDataclass(InnerDataclass):
    y: int


@pydantic.dataclasses.dataclass
class OuterDataclass:
    inner: InnerDataclass


d = OuterDataclass(inner=SubInnerDataclass(x=1, y=2))
print(d)
#> OuterDataclass(inner=SubInnerDataclass(x=1, y=2))
```

----------------------------------------

TITLE: JSON Serialization for Infinity and NaN
DESCRIPTION: Adds a mode to Pydantic-core for serializing JSON infinity and NaN values as strings ('Infinity', 'NaN') to ensure valid JSON output.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_40

LANGUAGE: APIDOC
CODE:
```
pydantic_core.PydanticCustomError(
  "JsonSchemaError",
  "Error serializing JSON schema: {error}",
  {
    "ser_json_inf_nan": "strings"
  }
)

# This setting affects how float('inf'), float('-inf'), and float('nan') are serialized.
# When 'strings' is used, they become "Infinity", "-Infinity", and "NaN" respectively.
# This is a pydantic-core setting, often configured via Pydantic's BaseModel.model_config.
```

----------------------------------------

TITLE: Pydantic v2.11.0b2 New Features
DESCRIPTION: Introduces experimental support for free threading in Pydantic v2.11.0b2.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_11

LANGUAGE: python
CODE:
```
# New Features in v2.11.0b2
# Add experimental support for free threading
```

----------------------------------------

TITLE: Pydantic V2 Serialization Customization
DESCRIPTION: Introduces new decorators for customizing serialization in Pydantic V2, replacing the older `json_encoders` config option. These decorators offer more granular control over how model fields and entire models are serialized.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_10

LANGUAGE: APIDOC
CODE:
```
Pydantic V2 Serialization Decorators:

- `@field_serializer(field_name, mode='wrap'|'plain'|'before'|'after')`
  - Purpose: Customize serialization for specific fields.
  - Parameters:
    - `field_name`: The name of the field to serialize.
    - `mode`: Controls when the serializer is called (e.g., 'wrap' for custom serialization logic, 'plain' for direct value return).
  - Usage: Apply to a method within the model class.

- `@model_serializer(mode='wrap'|'plain'|'before'|'after')`
  - Purpose: Customize serialization for the entire model instance.
  - Parameters:
    - `mode`: Controls when the serializer is called.
  - Usage: Apply to a method within the model class.

- `@computed_field`
  - Purpose: Define fields whose values are computed dynamically.
  - Usage: Apply to a method within the model class, which will be treated as a field.

Deprecated Feature:
- `json_encoders` in model config is deprecated due to performance and complexity. Use the new decorators instead.

Example Usage:
```python
from pydantic import BaseModel, field_serializer, computed_field
from datetime import datetime

class Event(BaseModel):
    name: str
    timestamp: datetime

    @field_serializer('timestamp', mode='iso')
    def serialize_timestamp(self, dt: datetime) -> str:
        return dt.isoformat()

    @computed_field
    @property
    def display_name(self) -> str:
        return f"Event: {self.name}"

# Example of model_serializer (not shown in original text, but implied by context)
# @model_serializer(mode='wrap')
# def serialize_model(self, handler):
#     data = handler(self)
#     return {"custom_data": data}


event = Event(name='Meeting', timestamp=datetime.now())
print(event.model_dump())
# Expected output might include: {'name': 'Meeting', 'timestamp': '2023-10-27T10:00:00.000000', 'display_name': 'Event: Meeting'}
```
```

----------------------------------------

TITLE: Serialize Pydantic Model to Dictionary
DESCRIPTION: Explains how to serialize a Pydantic model instance into a dictionary using the `model_dump()` method. This method is preferred over the built-in `dict()` for recursive conversion and offers customization options.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_5

LANGUAGE: python
CODE:
```
assert user.model_dump() == {'id': 123, 'name': 'Jane Doe'}
```

----------------------------------------

TITLE: Validate Dictionaries with dict[str, int]
DESCRIPTION: Shows how to use Pydantic's support for standard Python dictionaries with type hints. The `dict` type attempts to convert input into a dictionary with specified key and value types.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/standard_library_types.md#_snippet_17

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ValidationError


class Model(BaseModel):
    x: dict[str, int]


m = Model(x={'foo': 1})
print(m.model_dump())
# Expected output: {'x': {'foo': 1}}

try:
    Model(x={'foo': '1'})
except ValidationError as e:
    print(e)
    # Expected output:
    # 1 validation error for Model
    # x
    #   Input should be a valid dictionary [type=dict_type, input_value='test', input_type=str]
```

----------------------------------------

TITLE: Pydantic Handling of `typing.Hashable`
DESCRIPTION: Explains how Pydantic validates `typing.Hashable` types, distinguishing between Python native checks (using `isinstance`) and JSON input processing, where data is first loaded via an `Any` validator before checking for hashability.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/standard_library_types.md#_snippet_32

LANGUAGE: APIDOC
CODE:
```
typing.Hashable:
  From Python: Supports any data that passes an `isinstance(v, Hashable)` check.
  From JSON: First loads the data via an `Any` validator, then checks if the data is hashable with `isinstance(v, Hashable)`.
```

----------------------------------------

TITLE: Ignoring Specific Parameters in Argument Validation Schema
DESCRIPTION: Demonstrates how to customize argument schema generation by providing a `parameters_callback` to `generate_arguments_schema`. This callback function can specify which parameters to skip during schema creation, allowing for more flexible argument validation.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/experimental.md#_snippet_17

LANGUAGE: python
CODE:
```
from typing import Any

from pydantic_core import SchemaValidator

from pydantic.experimental.arguments_schema import generate_arguments_schema


def func(p: bool, *args: str, **kwargs: int) -> None: ...


def skip_first_parameter(index: int, name: str, annotation: Any) -> Any:
    if index == 0:
        return 'skip'


arguments_schema = generate_arguments_schema(
    func=func,
    parameters_callback=skip_first_parameter,
)

val = SchemaValidator(arguments_schema)

args, kwargs = val.validate_json('{"args": ["arg1"], "kwargs": {"extra": 1}}')
print(args, kwargs)
#> ('arg1',) {'extra': 1}
```

----------------------------------------

TITLE: Fixes and Enhancements
DESCRIPTION: A collection of fixes and minor enhancements addressing issues in JSON Schema generation, serialization behavior, exception messages, and handling of arbitrary schemas.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_16

LANGUAGE: python
CODE:
```
Use the correct JSON Schema mode when handling function schemas
Fix JSON Schema reference logic with `examples` keys
Improve exception message when encountering recursion errors during type evaluation
Always include `additionalProperties: True` for arbitrary dictionary schemas
Expose `fallback` parameter in serialization methods
Fix path serialization behavior
```

----------------------------------------

TITLE: Pydantic Model with Strict Boolean Field
DESCRIPTION: Illustrates defining a Pydantic `BaseModel` with a boolean field (`foo`) configured to be strictly validated using `Field(strict=True)`. This demonstrates how Python model definitions translate into core schema configurations.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/internals/architecture.md#_snippet_1

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field


class Model(BaseModel):
    foo: bool = Field(strict=True)
```

----------------------------------------

TITLE: Pydantic Structural Pattern Matching (PEP 636)
DESCRIPTION: Illustrates Pydantic's support for structural pattern matching as introduced in Python 3.10 (PEP 636). Pydantic models can be directly used in `match` statements for declarative data validation and attribute extraction.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_58

LANGUAGE: python
CODE:
```
from pydantic import BaseModel


class Pet(BaseModel):
    name: str
    species: str


a = Pet(name='Bones', species='dog')

match a:
    # match `species` to 'dog', declare and initialize `dog_name`
    case Pet(species='dog', name=dog_name):
        print(f'{dog_name} is a dog')
#> Bones is a dog
    # default case
    case _:
        print('No dog matched')
```

----------------------------------------

TITLE: Generate JSON Schema from Pydantic TypeAdapter
DESCRIPTION: Shows how to generate a JSON schema for arbitrary types using Pydantic's `TypeAdapter`. This method is a replacement for older schema generation functions and is useful for types not defined as Pydantic models.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/json_schema.md#_snippet_2

LANGUAGE: python
CODE:
```
from pydantic import TypeAdapter

adapter = TypeAdapter(list[int])
print(adapter.json_schema())
#> {'items': {'type': 'integer'}, 'type': 'array'}
```

----------------------------------------

TITLE: Custom Type Definition Hooks
DESCRIPTION: Details the replacement of Pydantic V1's `__get_validators__` with `__get_pydantic_core_schema__` and `__modify_schema__` with `__get_pydantic_json_schema__` for defining custom types and generating schemas.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_29

LANGUAGE: python
CODE:
```
# Pydantic V1 hook (deprecated)
# __get_validators__

# Pydantic V2 hooks
# __get_pydantic_core_schema__
# __get_pydantic_json_schema__
```

----------------------------------------

TITLE: Model Serializer - Wrap Mode
DESCRIPTION: Demonstrates the 'wrap' mode for @model_serializer, which provides more flexibility by allowing custom logic before or after Pydantic's default serialization. It requires a 'handler' parameter to delegate serialization.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/serialization.md#_snippet_12

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, SerializerFunctionWrapHandler, model_serializer


class UserModel(BaseModel):
    username: str
    password: str

    @model_serializer(mode='wrap')
    def serialize_model(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        serialized = handler(self)
        serialized['fields'] = list(serialized)
        return serialized


print(UserModel(username='foo', password='bar').model_dump())
#> {'username': 'foo', 'password': 'bar', 'fields': ['username', 'password']}
```

----------------------------------------

TITLE: Pydantic Annotated Pattern for Field Metadata
DESCRIPTION: Illustrates using the Annotated typing construct to attach metadata like Field() and WithJsonSchema to model fields.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/fields.md#_snippet_2

LANGUAGE: python
CODE:
```
from typing import Annotated

from pydantic import BaseModel, Field, WithJsonSchema


class Model(BaseModel):
    name: Annotated[str, Field(strict=True), WithJsonSchema({'extra': 'data'})]
```

----------------------------------------

TITLE: Default Factory with Callable
DESCRIPTION: Shows how to use `default_factory` with a lambda function to generate a default value, such as a UUID.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/fields.md#_snippet_7

LANGUAGE: python
CODE:
```
from uuid import uuid4

from pydantic import BaseModel, Field


class User(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
```

----------------------------------------

TITLE: Pydantic ValidationError Handling
DESCRIPTION: Demonstrates catching and printing Pydantic's `ValidationError` when data fails validation. The exception object contains details about all validation errors encountered.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_19

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ValidationError


class Model(BaseModel):
    list_of_ints: list[int]
    a_float: float


data = dict(
    list_of_ints=['1', 2, 'bad'],
    a_float='not a float',
)

try:
    Model(**data)
except ValidationError as e:
    print(e)
    # > 2 validation errors for Model
    # > list_of_ints.2
    # >   Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='bad', input_type=str]
    # > a_float
    # >   Input should be a valid number, unable to parse string as a number [type=float_parsing, input_value='not a float', input_type=str]
    # >

```

----------------------------------------

TITLE: Stdlib Dataclass Integration
DESCRIPTION: Facilitates seamless integration between standard library `dataclasses` and Pydantic's `dataclasses`. This includes converting standard library dataclasses into Pydantic dataclasses and allowing the use of standard library dataclasses within Pydantic models.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_194

LANGUAGE: python
CODE:
```
# Convert stdlib dataclasses to pydantic dataclasses
# Use stdlib dataclasses in models
```

----------------------------------------

TITLE: Handling Third-Party Types with Pydantic Annotations
DESCRIPTION: Illustrates how to integrate third-party classes into Pydantic models using Annotated types. It defines `__get_pydantic_core_schema__` and `__get_pydantic_json_schema__` to specify parsing, validation, and serialization logic for the external type. Dependencies include `pydantic`, `pydantic_core`, and `typing`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/types.md#_snippet_14

LANGUAGE: python
CODE:
```
from typing import Annotated, Any

from pydantic_core import core_schema

from pydantic import (
    BaseModel,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    ValidationError,
)
from pydantic.json_schema import JsonSchemaValue


class ThirdPartyType:
    """
    This is meant to represent a type from a third-party library that wasn't designed with Pydantic
    integration in mind, and so doesn't have a `pydantic_core.CoreSchema` or anything.
    """

    x: int

    def __init__(self):
        self.x = 0


class _ThirdPartyTypePydanticAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """
        We return a pydantic_core.CoreSchema that behaves in the following ways:

        * ints will be parsed as `ThirdPartyType` instances with the int as the x attribute
        * `ThirdPartyType` instances will be parsed as `ThirdPartyType` instances without any changes
        * Nothing else will pass validation
        * Serialization will always return just an int
        """

        def validate_from_int(value: int) -> ThirdPartyType:
            result = ThirdPartyType()
            result.x = value
            return result

        from_int_schema = core_schema.chain_schema(
            [
                core_schema.int_schema(),
                core_schema.no_info_plain_validator_function(validate_from_int),
            ]
        )

        return core_schema.json_or_python_schema(
            json_schema=from_int_schema,
            python_schema=core_schema.union_schema(
                [
                    # check if it's an instance first before doing any further work
                    core_schema.is_instance_schema(ThirdPartyType),
                    from_int_schema,
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: instance.x
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        # Use the same schema that would be used for `int`
        return handler(core_schema.int_schema())


# We now create an `Annotated` wrapper that we'll use as the annotation for fields on `BaseModel`s, etc.
PydanticThirdPartyType = Annotated[
    ThirdPartyType, _ThirdPartyTypePydanticAnnotation
]


# Create a model class that uses this annotation as a field
class Model(BaseModel):
    third_party_type: PydanticThirdPartyType


# Demonstrate that this field is handled correctly, that ints are parsed into `ThirdPartyType`, and that
# these instances are also "dumped" directly into ints as expected.

m_int = Model(third_party_type=1)
assert isinstance(m_int.third_party_type, ThirdPartyType)
assert m_int.third_party_type.x == 1
assert m_int.model_dump() == {'third_party_type': 1}
```

----------------------------------------

TITLE: RootModel with Generic Type for List
DESCRIPTION: Illustrates using `RootModel` with a generic type parameter to define a model that wraps a list of strings.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_47

LANGUAGE: python
CODE:
```
from pydantic import RootModel

Pets = RootModel[list[str]]
PetsByName = RootModel[dict[str, str]]


print(Pets(['dog', 'cat']))
#> root=['dog', 'cat']
print(Pets(['dog', 'cat']).model_dump_json())
#> ["dog","cat"]
print(Pets.model_validate(['dog', 'cat']))
#> root=['dog', 'cat']
print(Pets.model_json_schema())
"""
{'items': {'type': 'string'}, 'title': 'RootModel[list[str]]', 'type': 'array'}
"""

print(PetsByName({'Otis': 'dog', 'Milo': 'cat'}))
#> root={'Otis': 'dog', 'Milo': 'cat'}
print(PetsByName({'Otis': 'dog', 'Milo': 'cat'}).model_dump_json())
#> {"Otis":"dog","Milo":"cat"}
print(PetsByName.model_validate({'Otis': 'dog', 'Milo': 'cat'}))
#> root={'Otis': 'dog', 'Milo': 'cat'}
```

----------------------------------------

TITLE: Using Default Values with Partial JSON Parsing
DESCRIPTION: Demonstrates advanced partial JSON parsing by using `WrapValidator` with a custom error handler to provide default values for missing fields. This ensures that models with optional or defaultable fields can be validated even with incomplete JSON input.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/json.md#_snippet_4

LANGUAGE: python
CODE:
```
from typing import Annotated, Any, Optional

import pydantic_core

from pydantic import BaseModel, ValidationError, WrapValidator


def default_on_error(v, handler) -> Any:
    """
    Raise a PydanticUseDefault exception if the value is missing.

    This is useful for avoiding errors from partial
    JSON preventing successful validation.
    """
    try:
        return handler(v)
    except ValidationError as exc:
        # there might be other types of errors resulting from partial JSON parsing
        # that you allow here, feel free to customize as needed
        if all(e['type'] == 'missing' for e in exc.errors()):
            raise pydantic_core.PydanticUseDefault()
        else:
            raise


class NestedModel(BaseModel):
    x: int
    y: str


class MyModel(BaseModel):
    foo: Optional[str] = None
    bar: Annotated[
        Optional[tuple[str, int]], WrapValidator(default_on_error)
    ] = None
    nested: Annotated[
        Optional[NestedModel], WrapValidator(default_on_error)
    ] = None


m = MyModel.model_validate(
    pydantic_core.from_json('{"foo": "x", "bar": ["world",', allow_partial=True)
)
print(repr(m))
#> MyModel(foo='x', bar=None, nested=None)


m = MyModel.model_validate(
    pydantic_core.from_json(
        '{"foo": "x", "bar": ["world", 1], "nested": {"x":', allow_partial=True
    )
)
print(repr(m))
#> MyModel(foo='x', bar=('world', 1), nested=None)
```

----------------------------------------

TITLE: Pydantic GenerateJsonSchema API Reference
DESCRIPTION: Provides API documentation for the `pydantic.json_schema.GenerateJsonSchema` class. This class is the core mechanism for customizing JSON schema generation, allowing users to override specific methods to alter the schema output.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/json_schema.md#_snippet_29

LANGUAGE: APIDOC
CODE:
```
pydantic.json_schema.GenerateJsonSchema
  Description: Base class for customizing JSON schema generation.
  Methods: Implements methods for translating pydantic-core schema into JSON schema, designed for easy overriding in subclasses.
```

----------------------------------------

TITLE: Data Handling and Serialization
DESCRIPTION: Updates related to data parsing, serialization, and handling of specific data types like datetime and timedelta.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_296

LANGUAGE: APIDOC
CODE:
```
Datetime Parsing:
  - Fixed datetime parsing in `parse_date`.

Timedelta Serialization:
  - Timedelta JSON encoding supports ISO8601 and total seconds.
  - Custom JSON encoders for timedelta.

JSON Serialization:
  - JSON serialization of models and schemas.
```

----------------------------------------

TITLE: TypeAdapter for TypedDict List Validation and Serialization
DESCRIPTION: Demonstrates using TypeAdapter to validate and serialize lists of TypedDict objects. It covers handling validation errors and highlights that dump_json returns bytes, unlike BaseModel's model_dump_json which returns str.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/type_adapter.md#_snippet_0

LANGUAGE: python
CODE:
```
from typing_extensions import TypedDict

from pydantic import TypeAdapter, ValidationError


class User(TypedDict):
    name: str
    id: int


user_list_adapter = TypeAdapter(list[User])
user_list = user_list_adapter.validate_python([{'name': 'Fred', 'id': '3'}])
print(repr(user_list))
#> [{'name': 'Fred', 'id': 3}]

try:
    user_list_adapter.validate_python(
        [{'name': 'Fred', 'id': 'wrong', 'other': 'no'}]
    )
except ValidationError as e:
    print(e)
    """
    1 validation error for list[User]
    0.id
      Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='wrong', input_type=str]
    """

print(repr(user_list_adapter.dump_json(user_list)))
#> b'[{"name":"Fred","id":3}]'
```

----------------------------------------

TITLE: Handling Mutable Default Values
DESCRIPTION: Explains and demonstrates Pydantic's behavior with mutable default values, where it creates deep copies to ensure instance isolation.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/fields.md#_snippet_10

LANGUAGE: python
CODE:
```
from pydantic import BaseModel


class Model(BaseModel):
    item_counts: list[dict[str, int]] = [{}]


m1 = Model()
m1.item_counts[0]['a'] = 1
print(m1.item_counts)
#> [{'a': 1}]

m2 = Model()
print(m2.item_counts)
#> [{}]
```

----------------------------------------

TITLE: Pydantic v2.11.0 New Features
DESCRIPTION: Introduces new features in Pydantic v2.11.0, focusing on build time performance. Key additions include an `encoded_string()` method for URL types, support for `defer_build` with `@validate_call`, allowing `@with_config` with keyword arguments, simplifying default value inclusion in JSON Schema, and a new `generate_arguments_schema()` function.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_9

LANGUAGE: python
CODE:
```
# New Features in v2.11.0
# Add encoded_string() method to the URL types
# Add support for defer_build with @validate_call decorator
# Allow @with_config decorator to be used with keyword arguments
# Simplify customization of default value inclusion in JSON Schema generation
# Add generate_arguments_schema() function
```

----------------------------------------

TITLE: Pydantic Model String Representation
DESCRIPTION: Added in v0.2.0, the `to_string` method provides a way to generate a string representation of a Pydantic model. The `pretty` argument allows for formatted output, enhancing readability for debugging and display.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_309

LANGUAGE: APIDOC
CODE:
```
to_string(pretty: bool = False)
  Generates a string representation of the model.
  Parameters:
    pretty: If True, formats the output for better readability (e.g., with indentation).
  Returns: A string representing the model's data.
```

----------------------------------------

TITLE: Pydantic v0.31: Performance and Error Handling Improvements
DESCRIPTION: Introduces performance improvements by removing `change_exceptions` and altering how pydantic errors are constructed. Also includes fixes for `StrictFloat` and `StrictInt` classes, and improved handling of `None` and `Optional` validators.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_267

LANGUAGE: APIDOC
CODE:
```
Pydantic v0.31 (Pre-release/Related to v0.32 features) Notes:

- Add `StrictFloat` and `StrictInt` classes.
  - Related to issue [#799](https://github.com/pydantic/pydantic/pull/799).
  - Contributed by @DerRidda.
- Improve handling of `None` and `Optional`.
  - Replace `whole` with `each_item` (inverse meaning, default `False`) on validators.
  - Related to issue [#803](https://github.com/pydantic/pydantic/pull/803).
  - Contributed by @samuelcolvin.
- Add support for `Type[T]` type hints.
  - Related to issue [#807](https://github.com/pydantic/pydantic/pull/807).
  - Contributed by @timonbimon.
- Performance improvements from removing `change_exceptions`.
  - Change how pydantic errors are constructed.
  - Related to issue [#819](https://github.com/pydantic/pydantic/pull/819).
  - Contributed by @samuelcolvin.
- Fix error message when a `BaseModel`-type model field causes a `ValidationError`.
  - Related to issue [#820](https://github.com/pydantic/pydantic/pull/820).
  - Contributed by @dmontagu.
- Allow `getter_dict` on `Config`.
  - Modify `GetterDict` to be more like a `Mapping` object.
  - Related to issue [#821](https://github.com/pydantic/pydantic/pull/821).
  - Contributed by @samuelcolvin.
- Only check `TypeVar` param on base `GenericModel` class.
  - Related to issue [#842](https://github.com/pydantic/pydantic/pull/842).
  - Contributed by @zpencerq.
- Rename internal model attributes.
  - `Model._schema_cache` -> `Model.__schema_cache__`
  - `Model._json_encoder` -> `Model.__json_encoder__`
  - `Model._custom_root_type` -> `Model.__custom_root_type__`
  - Related to issue [#851](https://github.com/pydantic/pydantic/pull/851).
  - Contributed by @samuelcolvin.
```

----------------------------------------

TITLE: Performance: Evaluate FieldInfo Annotations Lazily
DESCRIPTION: Optimizes schema building by evaluating `FieldInfo` annotations only when necessary. This reduces overhead during the schema generation process.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_30

LANGUAGE: python
CODE:
```
from pydantic import Field

# This change improves performance by deferring annotation evaluation.
```

----------------------------------------

TITLE: Field Strictness Affecting Model Instantiation
DESCRIPTION: Shows that setting strict=True on a Field affects validation during direct model instantiation. It demonstrates that a field marked as strict will reject string inputs, while a field marked as strict=False will accept them.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/strict_mode.md#_snippet_8

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field, ValidationError


class Model(BaseModel):
    x: int = Field(strict=True)
    y: int = Field(strict=False)


try:
    Model(x='1', y='2')
except ValidationError as exc:
    print(exc)
    """
    1 validation error for Model
    x
      Input should be a valid integer [type=int_type, input_value='1', input_type=str]
    """

```

----------------------------------------

TITLE: Default Factory with Data Argument
DESCRIPTION: Illustrates using `default_factory` with a lambda that accepts the validated data dictionary as an argument to derive a default value.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/fields.md#_snippet_8

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, EmailStr, Field


class User(BaseModel):
    email: EmailStr
    username: str = Field(default_factory=lambda data: data['email'])


user = User(email='user@example.com')
print(user.username)
#> user@example.com
```

----------------------------------------

TITLE: Pydantic Alias Generator with Callable
DESCRIPTION: Shows how to use an alias generator with a callable (e.g., a lambda function) to automatically convert field names to uppercase for serialization. This simplifies consistent naming conventions across models.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/alias.md#_snippet_3

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ConfigDict


class Tree(BaseModel):
    model_config = ConfigDict(
        alias_generator=lambda field_name: field_name.upper()
    )

    age: int
    height: float
    kind: str


t = Tree.model_validate({'AGE': 12, 'HEIGHT': 1.2, 'KIND': 'oak'})
print(t.model_dump(by_alias=True))
#> {'AGE': 12, 'HEIGHT': 1.2, 'KIND': 'oak'}
```

----------------------------------------

TITLE: Constrained Strings as Dict Keys
DESCRIPTION: Added support for using constrained strings as dictionary keys in schema generation, allowing for more specific key validation.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_286

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, conint, constr

class ConstrainedKeyModel(BaseModel):
    # This example demonstrates the concept, actual key validation might differ
    # Pydantic primarily validates values, but schema generation reflects constraints
    data: dict[constr(min_length=3), int]
```

----------------------------------------

TITLE: Pydantic Breaking Changes and Behavior Updates
DESCRIPTION: Highlights significant changes and behavioral modifications in Pydantic releases. This includes updates to error formatting, argument behavior, and configuration handling.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_303

LANGUAGE: APIDOC
CODE:
```
Pydantic Breaking Changes and Behavior Updates:

Errors Format:
  - New errors format introduced, potentially breaking existing error handling.

conint/confloat arguments:
  - Corrected behavior for `lt` and `gt` arguments. Use `le` and `ge` for previous behavior.

Config attributes:
  - Removed `Config.min_number_size` and `Config.max_number_size`.
  - Renamed `.config` to `.__config__` on models.

Model values:
  - Defaults are now copied to model values, preventing shared mutable objects between models.

`.values()` method:
  - Deprecated, use `.dict()` instead.
```

----------------------------------------

TITLE: Support for Postponed Annotations
DESCRIPTION: Added support for postponed annotations, allowing type hints to refer to classes defined later in the same module or file.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_281

LANGUAGE: python
CODE:
```
from __future__ import annotations
from pydantic import BaseModel

class Node(BaseModel):
    next: 'Node' | None = None # Postponed annotation
```

----------------------------------------

TITLE: Support Generics Model with create_model
DESCRIPTION: Enables the use of generic types within models created dynamically using `pydantic.create_model`, expanding flexibility for dynamic model creation.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_86

LANGUAGE: python
CODE:
```
from pydantic import create_model
from typing import List, TypeVar

T = TypeVar('T')

# Dynamically create a model with a generic list field
GenericListModel = create_model('GenericListModel', items='List[T]')

# Usage:
# model_int = GenericListModel[int](items=[1, 2, 3])
# model_str = GenericListModel[str](items=['a', 'b'])

```

----------------------------------------

TITLE: Iterating Over Pydantic Models
DESCRIPTION: Demonstrates how to iterate over Pydantic models, yielding field names and values. Sub-models are not automatically converted to dictionaries during iteration. This functionality allows for easy inspection of model contents.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/serialization.md#_snippet_2

LANGUAGE: python
CODE:
```
from pydantic import BaseModel


class BarModel(BaseModel):
    whatever: int


class FooBarModel(BaseModel):
    banana: float
    foo: str
    bar: BarModel


m = FooBarModel(banana=3.14, foo='hello', bar={'whatever': 123})

for name, value in m:
    print(f'{name}: {value}')
    #> banana: 3.14
    #> foo: hello
    #> bar: whatever=123
```

LANGUAGE: python
CODE:
```
print(dict(m))
#> {'banana': 3.14, 'foo': 'hello', 'bar': BarModel(whatever=123)}
```

----------------------------------------

TITLE: Validating Default Values
DESCRIPTION: Demonstrates how to enable validation of default values using `validate_default=True` in `Field`, showing error handling for invalid defaults.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/fields.md#_snippet_9

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field, ValidationError


class User(BaseModel):
    age: int = Field(default='twelve', validate_default=True)


try:
    user = User()
except ValidationError as e:
    print(e)
    """
1 validation error for User
age
  Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='twelve', input_type=str]
"""
```

----------------------------------------

TITLE: ConfigDict Validation Settings
DESCRIPTION: Details configuration options within `ConfigDict` for controlling alias usage during validation. `validate_by_alias` (default True) and `validate_by_name` (default False) determine how Pydantic matches input data to model fields.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/alias.md#_snippet_10

LANGUAGE: APIDOC
CODE:
```
ConfigDict Settings for Validation:
  - validate_by_alias: Controls if aliases are used for validation. Defaults to True.
  - validate_by_name: Controls if attribute names are used for validation. Defaults to False.

  Note: You cannot set both `validate_by_alias` and `validate_by_name` to `False`. Doing so raises a user error.
```

----------------------------------------

TITLE: Automatic Pydantic Validation Instrumentation
DESCRIPTION: Shows how to automatically instrument Pydantic models with Logfire to capture validation success and failure events. This provides detailed insights into data validation processes within your application.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/integrations/logfire.md#_snippet_1

LANGUAGE: python
CODE:
```
from datetime import date

import logfire

from pydantic import BaseModel

logfire.configure()
logfire.instrument_pydantic()  # (1)!


class User(BaseModel):
    name: str
    country_code: str
    dob: date


User(name='Anne', country_code='USA', dob='2000-01-01')
User(name='David', country_code='GBR', dob='invalid-dob')
```

----------------------------------------

TITLE: Pydantic: Correct validator fields as arguments
DESCRIPTION: Shows the correct method for applying a validator to multiple fields by passing each field name as a separate string argument to the `@field_validator` decorator. This avoids the 'invalid validator fields' error.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/usage_errors.md#_snippet_29

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, field_validator


class Model(BaseModel):
    a: str
    b: str

    @field_validator('a', 'b')
    def check_fields(cls, v):
        return v
```

----------------------------------------

TITLE: Support for Callable Type Hint
DESCRIPTION: Introduced support for the `Callable` type hint, enhancing the library's ability to handle function signatures and callables.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_268

LANGUAGE: python
CODE:
```
from typing import Callable

def process_callback(callback: Callable[[int], str]):
    # ... implementation ...
```

----------------------------------------

TITLE: Pydantic Model from Nested Arbitrary Instances
DESCRIPTION: Illustrates how Pydantic models can parse nested arbitrary class instances when `from_attributes=True` is enabled. It shows parsing a person object containing a list of pet objects.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_18

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ConfigDict


class PetCls:
    def __init__(self, *, name: str, species: str):
        self.name = name
        self.species = species


class PersonCls:
    def __init__(self, *, name: str, age: float = None, pets: list[PetCls]):
        self.name = name
        self.age = age
        self.pets = pets


class Pet(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    species: str


class Person(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    age: float = None
    pets: list[Pet]


bones = PetCls(name='Bones', species='dog')
orion = PetCls(name='Orion', species='cat')
anna = PersonCls(name='Anna', age=20, pets=[bones, orion])
anna_model = Person.model_validate(anna)
print(anna_model)
# > name='Anna' age=20.0 pets=[Pet(name='Bones', species='dog'), Pet(name='Orion', species='cat')]

```

----------------------------------------

TITLE: Defining a Root Model in Pydantic
DESCRIPTION: Demonstrates how to define a basic Root Model in Pydantic. Root Models are useful when you need to validate a single value, such as a list or a primitive type, rather than a dictionary-like structure.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/root_model.md#_snippet_0

LANGUAGE: python
CODE:
```
from pydantic import RootModel

class MyListModel(RootModel[list[int]]):
    pass

# Example usage:
my_list = MyListModel([1, 2, 3])
print(my_list.root)
# Output: [1, 2, 3]

class MyStringModel(RootModel[str]):
    pass

my_string = MyStringModel("hello")
print(my_string.root)
# Output: hello
```

----------------------------------------

TITLE: Pydantic AliasPath for Nested Field Aliases
DESCRIPTION: Demonstrates using AliasPath to specify a path to a field, including array indices, for validation. Shows how to access nested data using aliases.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/alias.md#_snippet_0

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field, AliasPath


class User(BaseModel):
    first_name: str = Field(validation_alias=AliasPath('names', 0))
    last_name: str = Field(validation_alias=AliasPath('names', 1))

user = User.model_validate({'names': ['John', 'Doe']})  # (1)!
print(user)
#> first_name='John' last_name='Doe'
```

----------------------------------------

TITLE: Validate JSON with Multiple Partial Items
DESCRIPTION: Demonstrates validating a JSON string containing multiple items, where one item is complete and valid, and another is incomplete. Partial validation correctly processes the complete item and handles the incomplete one.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/experimental.md#_snippet_8

LANGUAGE: python
CODE:
```
from typing import Annotated

from annotated_types import MinLen
from typing_extensions import NotRequired, TypedDict

from pydantic import TypeAdapter


class Foobar(TypedDict):
    a: int
    b: NotRequired[float]
    c: NotRequired[Annotated[str, MinLen(5)]]


ta = TypeAdapter(list[Foobar])

v = ta.validate_json(
    '[{"a": 1, "b": 1.0, "c": "abcde"},{"a": ', experimental_allow_partial=True
)
print(v)
#> [{'a': 1, 'b': 1.0, 'c': 'abcde'}]
```

----------------------------------------

TITLE: MongoDB DSN Schema Support
DESCRIPTION: Adds a schema for validating MongoDB network data source names (DSNs), enabling robust configuration for MongoDB connections.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_106

LANGUAGE: python
CODE:
```
from pydantic import BaseModel
from pydantic.networks import MongoDsn

class MongoConfig(BaseModel):
    # Validates MongoDB connection strings
    db_uri: MongoDsn

# Example:
# config = MongoConfig(db_uri='mongodb://user:pass@host:port/database?authSource=admin')
# Invalid DSNs will raise a ValidationError.
```

----------------------------------------

TITLE: Avoid with_config on BaseModel Subclasses
DESCRIPTION: Demonstrates the PydanticUserError raised when the `with_config` decorator is incorrectly applied to a class that already inherits from `BaseModel`. The correct approach is to use the `model_config` attribute.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/usage_errors.md#_snippet_49

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, PydanticUserError, with_config

try:

    @with_config({'allow_inf_nan': True})
    class Model(BaseModel):
        bar: str

except PydanticUserError as exc_info:
    assert exc_info.code == 'with-config-on-model'
```

----------------------------------------

TITLE: Model Serializer - Plain Mode
DESCRIPTION: Shows how to use the @model_serializer decorator in 'plain' mode to serialize an entire Pydantic model into a custom string format. The 'plain' mode is the default and allows returning non-dictionary values.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/serialization.md#_snippet_11

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, model_serializer


class UserModel(BaseModel):
    username: str
    password: str

    @model_serializer(mode='plain')  # (1)!
    def serialize_model(self) -> str:  # (2)!
        return f'{self.username} - {self.password}'


print(UserModel(username='foo', password='bar').model_dump())
#> foo - bar
```

----------------------------------------

TITLE: Pydantic Schema Generation: Pattern Type
DESCRIPTION: Adds basic support for the `Pattern` type in schema generation, allowing regular expressions to be included in OpenAPI/JSON schemas.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_205

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field
from typing import Pattern
import re

class RegexModel(BaseModel):
    # Using Field with Pattern type
    uuid_string: Pattern[str] = Field(regex=r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')

# When generating schema for RegexModel, the regex pattern will be included.
# Example: RegexModel(uuid_string='123e4567-e89b-12d3-a456-426614174000')

```

----------------------------------------

TITLE: Generic Subclassing of Pydantic Models
DESCRIPTION: Illustrates creating a generic subclass of a Pydantic model that inherits and potentially redefines type variables from its superclass. This allows for flexible type hinting and model definition.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_28

LANGUAGE: python
CODE:
```
from typing import Generic, TypeVar

from pydantic import BaseModel

TypeX = TypeVar('TypeX')
TypeY = TypeVar('TypeY')
TypeZ = TypeVar('TypeZ')


class BaseClass(BaseModel, Generic[TypeX, TypeY]):
    x: TypeX
    y: TypeY


class ChildClass(BaseClass[int, TypeY], Generic[TypeY, TypeZ]):
    z: TypeZ
```

----------------------------------------

TITLE: SecretsSettingsSource Respects Case Sensitivity
DESCRIPTION: Updates `SecretsSettingsSource` to correctly respect the `config.case_sensitive` setting, ensuring environment variable lookup is case-aware as configured.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_105

LANGUAGE: python
CODE:
```
from pydantic import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):
    # If case_sensitive is True, MY_VAR is different from my_var
    my_var: str

    model_config = SettingsConfigDict(env_prefix='APP_', case_sensitive=True)

# Example:
# If env var is MY_VAR='value', and case_sensitive=True, it will be loaded.
# If case_sensitive=False, my_var='value' would also load MY_VAR.
```

----------------------------------------

TITLE: validate_call: Positional or Keyword Parameters
DESCRIPTION: Demonstrates the `validate_call` decorator with functions accepting positional or keyword parameters, including those with default values. It shows how arguments are passed and validated against type hints.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/validation_decorator.md#_snippet_2

LANGUAGE: python
CODE:
```
from pydantic import validate_call


@validate_call
def pos_or_kw(a: int, b: int = 2) -> str:
    return f'a={a} b={b}'


print(pos_or_kw(1, b=3))
#> a=1 b=3
```

----------------------------------------

TITLE: Pydantic: Frozen Models for Hashability
DESCRIPTION: Introduces a `frozen=True` parameter in `Config`. Setting this makes models immutable and generates a `__hash__()` method, enabling instances to be hashable if their attributes are hashable.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_183

LANGUAGE: python
CODE:
```
from pydantic import BaseModel

class ImmutableModel(BaseModel):
    class Config:
        frozen = True

    id: int
    name: str

# Example usage:
# model_instance = ImmutableModel(id=1, name='Example')
# print(hash(model_instance))
# 
# # This would raise a TypeError:
# # model_instance.id = 2
```

----------------------------------------

TITLE: Pydantic User Model Validation
DESCRIPTION: Defines a Pydantic BaseModel 'User' with various field types and demonstrates how to catch and print validation errors when instantiating the model with incorrect data.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/index.md#_snippet_2

LANGUAGE: python
CODE:
```
from datetime import datetime
from pydantic import BaseModel, PositiveInt, ValidationError


class User(BaseModel):
    id: int
    name: str = 'John Doe'
    signup_ts: datetime | None
    tastes: dict[str, PositiveInt]


external_data = {'id': 'not an int', 'tastes': {}}  # (1)!

try:
    User(**external_data)  # (2)!
except ValidationError as e:
    print(e.errors())
```

LANGUAGE: python
CODE:
```
[
    {
        'type': 'int_parsing',
        'loc': ('id',),
        'msg': 'Input should be a valid integer, unable to parse string as an integer',
        'input': 'not an int',
        'url': 'https://errors.pydantic.dev/2/v/int_parsing',
    },
    {
        'type': 'missing',
        'loc': ('signup_ts',),
        'msg': 'Field required',
        'input': {'id': 'not an int', 'tastes': {}},
        'url': 'https://errors.pydantic.dev/2/v/missing',
    },
]
```

----------------------------------------

TITLE: Pydantic Type Handling for Standard Library
DESCRIPTION: Provides an overview of Pydantic's validation and coercion mechanisms for core Python types like int, float, and enum.IntEnum, detailing accepted inputs and conversion processes.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/standard_library_types.md#_snippet_5

LANGUAGE: APIDOC
CODE:
```
Pydantic Standard Library Type Support:

Booleans:
  - Accepts: bool, int (0/1), str ('true', 'false', 'yes', 'no', etc. case-insensitive), bytes (decoded to str).
  - StrictBool: For only accepting True/False.

Datetimes:
  datetime:
    - Accepts: datetime object, int/float (Unix time in seconds or milliseconds), str (ISO 8601 format, YYYY-MM-DD).
    - date objects are accepted in lax mode.
  date:
    - Accepts: date object, int/float (Unix time), str (YYYY-MM-DD).
  time:
    - Accepts: time object, str (HH:MM[:SS[.ffffff]]).
  timedelta:
    - Accepts: timedelta object, int/float (seconds), str (e.g., '1d,01:02:03', ISO 8601 P...T...S format).

Numbers:
  int:
    - Coerces using int(v).
  float:
    - Coerces using float(v).

Enums:
  enum.IntEnum:
    - Validates that the value is a valid IntEnum instance or member.

```

----------------------------------------

TITLE: Dataclass InitVar Serialization
DESCRIPTION: Correction to ensure `dataclass` `InitVar` is not required during serialization, aligning with expected behavior.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_19

LANGUAGE: python
CODE:
```
`dataclass` `InitVar` shouldn't be required on serialization
```

----------------------------------------

TITLE: Comparison Method for Color Class
DESCRIPTION: Adds a comparison method to the `Color` class, enabling direct comparison of color instances based on their properties.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_97

LANGUAGE: python
CODE:
```
from pydantic import BaseModel

class Color(BaseModel):
    r: int
    g: int
    b: int

    def __eq__(self, other):
        if not isinstance(other, Color):
            return NotImplemented
        return self.r == other.r and self.g == other.g and self.b == other.b

# Example:
# color1 = Color(r=255, g=0, b=0)
# color2 = Color(r=255, g=0, b=0)
# print(color1 == color2) # True
```

----------------------------------------

TITLE: Config.anystr_upper and constr/conbytes to_upper
DESCRIPTION: Introduces `Config.anystr_upper` and a `to_upper` keyword argument for `constr` (constrained string) and `conbytes` (constrained bytes). These allow for automatic conversion of string or byte values to uppercase during validation.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_76

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field, constr

class UpperCaseModel(BaseModel):
    class Config:
        anystr_upper = True

    name: str
    code: constr(to_upper=True)

model = UpperCaseModel(name="test", code="abc")

print(model.name) # Output: TEST
print(model.code) # Output: ABC

```

----------------------------------------

TITLE: Performance: Speed up _get_all_json_refs()
DESCRIPTION: Boosts the performance of `_get_all_json_refs()` by 34% in `pydantic/json_schema.py`, optimizing JSON schema generation.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_46

LANGUAGE: python
CODE:
```
# Internal optimization in pydantic/json_schema.py
# Improves the efficiency of resolving and collecting JSON references within schemas.
```

----------------------------------------

TITLE: Pydantic Field Customization with typing.Annotated
DESCRIPTION: Shows how to specify JSON schema modifications via the Field constructor using typing.Annotated. This approach allows for cleaner integration of field metadata and constraints directly within the type hint.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/json_schema.md#_snippet_7

LANGUAGE: python
CODE:
```
import json
from typing import Annotated
from uuid import uuid4

from pydantic import BaseModel, Field


class Foo(BaseModel):
    id: Annotated[str, Field(default_factory=lambda: uuid4().hex)]
    name: Annotated[str, Field(max_length=256)] = Field(
        'Bar', title='CustomName'
    )


print(json.dumps(Foo.model_json_schema(), indent=2))
```

LANGUAGE: json
CODE:
```
{
  "properties": {
    "id": {
      "title": "Id",
      "type": "string"
    },
    "name": {
      "default": "Bar",
      "maxLength": 256,
      "title": "CustomName",
      "type": "string"
    }
  },
  "title": "Foo",
  "type": "object"
}
```

----------------------------------------

TITLE: Pydantic Custom Cyclic Reference Handling
DESCRIPTION: Provides a custom validator and context manager to gracefully handle and suppress `ValidationError` instances caused by cyclic references, allowing for controlled processing of such data structures.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/forward_annotations.md#_snippet_3

LANGUAGE: python
CODE:
```
from contextlib import contextmanager
from dataclasses import field
from typing import Iterator

from pydantic import BaseModel, ValidationError, field_validator


def is_recursion_validation_error(exc: ValidationError) -> bool:
    errors = exc.errors()
    return len(errors) == 1 and errors[0]['type'] == 'recursion_loop'


@contextmanager
def suppress_recursion_validation_error() -> Iterator[None]:
    try:
        yield
    except ValidationError as exc:
        if not is_recursion_validation_error(exc):
            raise exc


class Node(BaseModel):
    id: int
    children: list['Node'] = field(default_factory=list)

    @field_validator('children', mode='wrap')
    @classmethod
    def drop_cyclic_references(cls, children, h):
        try:
            return h(children)
        except ValidationError as exc:
            if not (
                is_recursion_validation_error(exc)
                and isinstance(children, list)
            ):
                raise exc

            value_without_cyclic_refs = []
            for child in children:
                with suppress_recursion_validation_error():
                    value_without_cyclic_refs.extend(h([child]))
            return h(value_without_cyclic_refs)


# Create data with cyclic references representing the graph 1 -> 2 -> 3 -> 1
node_data = {'id': 1, 'children': [{'id': 2, 'children': [{'id': 3}]}]}
node_data['children'][0]['children'][0]['children'] = [node_data]

print(Node.model_validate(node_data))
#> id=1 children=[Node(id=2, children=[Node(id=3, children=[])])]
```

----------------------------------------

TITLE: Update AnyClassMethod for typeshed changes
DESCRIPTION: Updates the `AnyClassMethod` definition to align with recent changes in python/typeshed issue 9771. This ensures Pydantic's type hinting remains accurate and compatible.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_55

LANGUAGE: python
CODE:
```
# update `AnyClassMethod` for changes in [python/typeshed#9771](https://github.com/python/typeshed/issues/9771), [#5505](https://github.com/pydantic/pydantic/pull/5505) by @ITProKyle
```

----------------------------------------

TITLE: Introduction of conset() Validator
DESCRIPTION: Introduces `conset()`, a new validator analogous to `conlist()`, enabling constrained sets with specific validation rules. This expands Pydantic's capabilities for validating collection types.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_221

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, conset

class MyModel(BaseModel):
    my_set: conset(min_items=2, max_items=5, unique_items=True)

# Example usage:
# model = MyModel(my_set={1, 2, 3})
# model_invalid = MyModel(my_set={1})
# This would raise a ValidationError because min_items is 2
```

----------------------------------------

TITLE: Pydantic SQLAlchemy Model Integration
DESCRIPTION: Demonstrates integrating Pydantic models with SQLAlchemy by defining a Pydantic model that validates SQLAlchemy model instances. It highlights the use of Pydantic's `Field` with `alias` to map to SQLAlchemy columns, especially when column names conflict with Python reserved keywords or SQLAlchemy attributes. This approach helps manage database schema definitions with Pydantic's validation capabilities.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/examples/orms.md#_snippet_0

LANGUAGE: python
CODE:
```
import sqlalchemy as sa
from sqlalchemy.orm import declarative_base

from pydantic import BaseModel, ConfigDict, Field


class MyModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    metadata: dict[str, str] = Field(alias='metadata_')


Base = declarative_base()


class MyTableModel(Base):
    __tablename__ = 'my_table'
    id = sa.Column('id', sa.Integer, primary_key=True)
    # 'metadata' is reserved by SQLAlchemy, hence the '_'
    metadata_ = sa.Column('metadata', sa.JSON)


sql_model = MyTableModel(metadata_={'key': 'val'}, id=1)
pydantic_model = MyModel.model_validate(sql_model)

print(pydantic_model.model_dump())
#> {'metadata': {'key': 'val'}}
print(pydantic_model.model_dump(by_alias=True))
#> {'metadata_': {'key': 'val'}}
```

----------------------------------------

TITLE: Hypothesis plugin for constrained floats
DESCRIPTION: Enables the Hypothesis plugin to generate constrained floats when the `multiple_of` argument is specified. This improves property-based testing for numerical constraints.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_136

LANGUAGE: python
CODE:
```
enable the Hypothesis plugin to generate a constrained float when the `multiple_of` argument is specified, [#2442](https://github.com/pydantic/pydantic/pull/2442) by @tobi-lipede-oodle
```

----------------------------------------

TITLE: Pydantic TypeAdapter Partial Validation API
DESCRIPTION: Details the 'experimental_allow_partial' flag for Pydantic's TypeAdapter validation methods (validate_json, validate_python, validate_strings). It supports 'off', 'on', and 'trailing-strings' modes for handling incomplete data, with 'trailing-strings' allowing incomplete final strings.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/experimental.md#_snippet_4

LANGUAGE: APIDOC
CODE:
```
TypeAdapter.validate_json(..., experimental_allow_partial: bool | str = False)
TypeAdapter.validate_python(..., experimental_allow_partial: bool | str = False)
TypeAdapter.validate_strings(..., experimental_allow_partial: bool | str = False)

Parameters:
  experimental_allow_partial: Controls partial validation behavior.
    - False or 'off': Disables partial validation (default).
    - True or 'on': Enables partial validation, does not support trailing strings.
    - 'trailing-strings': Enables partial validation and supports trailing incomplete strings.

Description:
  Enables validation of incomplete JSON or Python data structures. Useful for streaming data where the input may be truncated.
  The 'trailing-strings' mode specifically allows incomplete string values at the end of the input to be included in the validated output.
```

----------------------------------------

TITLE: Pydantic parse_raw_as Utility
DESCRIPTION: Adds a new utility function `parse_raw_as` for parsing raw data into a specified Pydantic model or type.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_203

LANGUAGE: python
CODE:
```
from pydantic import parse_raw_as, BaseModel

class User(BaseModel):
    name: str
    age: int

json_data = '{"name": "Alice", "age": 30}'

# Parse JSON string into a User model
user_instance = parse_raw_as(User, json_data)
print(user_instance)

# Parse JSON string into a list of User models
json_list_data = '[{"name": "Bob", "age": 25}, {"name": "Charlie", "age": 35}]'
user_list = parse_raw_as(list[User], json_list_data)
print(user_list)
```

----------------------------------------

TITLE: Pydantic Model Validator (Wrap Mode)
DESCRIPTION: Illustrates a 'wrap' model validator in Pydantic, offering the most flexibility. It allows executing code before and after Pydantic's validation, or terminating validation early by returning data or raising an error.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/validators.md#_snippet_14

LANGUAGE: python
CODE:
```
import logging
from typing import Any

from typing_extensions import Self

from pydantic import BaseModel, ModelWrapValidatorHandler, ValidationError, model_validator


class UserModel(BaseModel):
    username: str

    @model_validator(mode='wrap')
    @classmethod
    def log_failed_validation(cls, data: Any, handler: ModelWrapValidatorHandler[Self]) -> Self:
        try:
            return handler(data)
        except ValidationError:
            logging.error('Model %s failed to validate with data %s', cls, data)
            raise
```

----------------------------------------

TITLE: Validate TypedDict with Pydantic
DESCRIPTION: Explains and demonstrates using Python's TypedDict with Pydantic for dictionaries with fixed keys and value types. Requires `typing-extensions` for Python versions prior to 3.12.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/standard_library_types.md#_snippet_18

LANGUAGE: python
CODE:
```
from typing_extensions import TypedDict

from pydantic import TypeAdapter, ValidationError


class User(TypedDict):
    name: str
    id: int


ta = TypeAdapter(User)

print(ta.validate_python({'name': 'foo', 'id': 1}))
# Expected output: {'name': 'foo', 'id': 1}

try:
    ta.validate_python({'name': 'foo'})
except ValidationError as e:
    print(e)
    # Expected output:
    # 1 validation error for User
    # id
    #   Field required [type=missing, input_value={'name': 'foo'}, input_type=dict]
```

----------------------------------------

TITLE: RootModel Extra Config Not Allowed in Pydantic
DESCRIPTION: This error occurs when `model_config['extra']` is specified with `RootModel`. `RootModel` does not support extra fields during initialization, making this configuration invalid.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/usage_errors.md#_snippet_44

LANGUAGE: python
CODE:
```
from pydantic import PydanticUserError, RootModel

try:

    class MyRootModel(RootModel):
        model_config = {'extra': 'allow'}
        root: int

except PydanticUserError as exc_info:
    assert exc_info.code == 'root-model-extra'
```

----------------------------------------

TITLE: Schema Generation and Validation
DESCRIPTION: Improvements and additions related to schema generation, compatibility with JSON Schema and OpenAPI, and validation for string and numeric types.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_293

LANGUAGE: APIDOC
CODE:
```
Schema Generation:
  - Refactored schema generation for compatibility with JSON Schema and OpenAPI specs.
  - Added `schema` to `schema` module for generating top-level schemas from base models.
  - Introduced `Schema` class with additional fields for declaring validation for `str` and numeric values.
  - Fixed schema generation for fields defined using `typing.Any`.
  - Major improvements and changes to schema generation.
  - Model schema generation.

Field Renaming:
  - Renamed `_schema` to `schema` on fields.

Validation Enhancements:
  - Fixed issue where `int_validator` did not cast a `bool` to an `int`.
  - Made `list`, `tuple`, and `set` types stricter.
  - Fixed `list`, `set`, & `tuple` validation.
  - Separated `validate_model` method to allow returning errors along with valid values.
```

----------------------------------------

TITLE: Pydantic Forward Refs and Optional Fields
DESCRIPTION: Fixes behavior with forward references and optional fields in nested Pydantic models.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_210

LANGUAGE: python
CODE:
```
from pydantic import BaseModel
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Forward reference to a model defined later or in another module
    class ParentModel(BaseModel):
        child: Optional['ChildModel']
else:
    # Actual definition for runtime
    class ChildModel(BaseModel):
        value: int

    class ParentModel(BaseModel):
        child: Optional[ChildModel]

# This ensures that optional fields with forward references are handled correctly.
```

----------------------------------------

TITLE: Pydantic Handling of `typing.Annotated`
DESCRIPTION: Details Pydantic's support for `typing.Annotated` as per PEP-593, noting that it allows arbitrary metadata but typically only processes a single call to the `Field` function, ignoring other additional metadata.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/standard_library_types.md#_snippet_33

LANGUAGE: APIDOC
CODE:
```
typing.Annotated:
  Description: Allows wrapping another type with arbitrary metadata, as per [PEP-593](https://www.python.org/dev/peps/pep-0593/).
  Behavior: The `Annotated` hint may contain a single call to the `Field` function, but otherwise the additional metadata is ignored and the root type is used.
```

----------------------------------------

TITLE: Pydantic Literal with Union and ClassVar
DESCRIPTION: Shows how typing.Literal can be combined with Union and ClassVar to create discriminated unions and define class-specific constants. Illustrates parsing different types within a union based on literal field values.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/standard_library_types.md#_snippet_26

LANGUAGE: python
CODE:
```
from typing import ClassVar, Literal, Union

from pydantic import BaseModel, ValidationError


class Cake(BaseModel):
    kind: Literal['cake']
    required_utensils: ClassVar[list[str]] = ['fork', 'knife']


class IceCream(BaseModel):
    kind: Literal['icecream']
    required_utensils: ClassVar[list[str]] = ['spoon']


class Meal(BaseModel):
    dessert: Union[Cake, IceCream]


print(type(Meal(dessert={'kind': 'cake'}).dessert).__name__)
print(type(Meal(dessert={'kind': 'icecream'}).dessert).__name__)
try:
    Meal(dessert={'kind': 'pie'})
except ValidationError as e:
    print(str(e))
    
```

----------------------------------------

TITLE: Python: Serialize Unparametrized Type Variables
DESCRIPTION: Details serialization differences for type variables with bounds or defaults when unparametrized. Shows how Pydantic treats values as `Any` for serialization when the generic type is not explicitly parametrized.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_37

LANGUAGE: python
CODE:
```
from typing import Generic, TypeVar

from pydantic import BaseModel


class ErrorDetails(BaseModel):
    foo: str


ErrorDataT = TypeVar('ErrorDataT', bound=ErrorDetails)


class Error(BaseModel, Generic[ErrorDataT]):
    message: str
    details: ErrorDataT


class MyErrorDetails(ErrorDetails):
    bar: str


# serialized as Any
error = Error(
    message='We just had an error',
    details=MyErrorDetails(foo='var', bar='var2'),
)
print(error.model_dump())
```

----------------------------------------

TITLE: Validate YAML Data with Pydantic in Python
DESCRIPTION: Validates data from a YAML file using Pydantic. It utilizes the `yaml` library to load YAML content and then validates the data against a Pydantic model. Requires `pydantic` and `PyYAML` libraries.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/examples/files.md#_snippet_6

LANGUAGE: python
CODE:
```
import yaml

from pydantic import BaseModel, EmailStr, PositiveInt


class Person(BaseModel):
    name: str
    age: PositiveInt
    email: EmailStr


with open('person.yaml') as f:
    data = yaml.safe_load(f)

person = Person.model_validate(data)
print(person)
#> name='John Doe' age=30 email='john@example.com'
```

----------------------------------------

TITLE: Inherit from Stdlib Dataclasses with Pydantic
DESCRIPTION: Demonstrates how Pydantic automatically validates fields when inheriting from standard library dataclasses. Shows how Pydantic's dataclass decorator handles nested inheritance and type coercion.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/dataclasses.md#_snippet_3

LANGUAGE: python
CODE:
```
import dataclasses

import pydantic


@dataclasses.dataclass
class Z:
    z: int


@dataclasses.dataclass
class Y(Z):
    y: int = 0


@pydantic.dataclasses.dataclass
class X(Y):
    x: int = 0


foo = X(x=b'1', y='2', z='3')
print(foo)
#> X(z=3, y=2, x=1)

try:
    X(z='pika')
except pydantic.ValidationError as e:
    print(e)
    """
    1 validation error for X
    z
      Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='pika', input_type=str]
    """

```

----------------------------------------

TITLE: Programmatic Title Generation
DESCRIPTION: Enables programmatic generation of titles for Pydantic models, enhancing schema clarity and documentation.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_38

LANGUAGE: APIDOC
CODE:
```
BaseModel.model_json_schema(by_alias: bool = True, ...):
  # ... schema generation logic ...
  # The 'title' field in the schema can be programmatically set.
  
  # Example:
  # class MyModel(BaseModel):
  #   class Config:
  #     title = "My Custom Model Title"
  #   field: int
```

----------------------------------------

TITLE: Configure Strict Mode with TypeAdapter in Pydantic
DESCRIPTION: Demonstrates how to use Pydantic's TypeAdapter with a ConfigDict to enable strict mode. Strict mode enforces that input values must conform precisely to the target type, rejecting coerced values. This snippet shows how to catch validation errors when non-strict inputs are provided.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/strict_mode.md#_snippet_17

LANGUAGE: python
CODE:
```
from pydantic import ConfigDict, TypeAdapter, ValidationError

adapter = TypeAdapter(bool, config=ConfigDict(strict=True))

try:
    adapter.validate_python('yes')
except ValidationError as exc:
    print(exc)
    
```

----------------------------------------

TITLE: Pydantic v2.11.1 Fixes
DESCRIPTION: Addresses an issue in Pydantic v2.11.1 where `'definitions-ref'` schemas containing serialization schemas or metadata were not handled correctly.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_8

LANGUAGE: python
CODE:
```
# Fixes in v2.11.1
# Do not override 'definitions-ref' schemas containing serialization schemas or metadata
```

----------------------------------------

TITLE: Pydantic BaseModel.model_copy API
DESCRIPTION: API documentation for the `model_copy()` method of Pydantic's `BaseModel`. This method allows for creating a copy of a model instance, optionally updating fields or performing a deep copy.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_24

LANGUAGE: APIDOC
CODE:
```
pydantic.main.BaseModel.model_copy

Copies the model instance.

Parameters:
  update (Optional[Dict[str, Any]]): A dictionary of fields to update in the new model instance. Defaults to None.
  deep (bool): If True, perform a deep copy of the model instance. Defaults to False.

Returns:
  BaseModel: A new model instance that is a copy of the original, with optional updates and deep copy behavior.

Notes:
  - Useful for working with frozen models.
  - When `deep=True`, nested models and mutable objects (like lists or dicts) are also copied recursively.
  - When `deep=False` (default), nested models and mutable objects are shared by reference between the original and the copied instance.
```

----------------------------------------

TITLE: Config Field Defaults for String Length
DESCRIPTION: Changed default values for `BaseConfig` attributes `min_anystr_length` and `max_anystr_length` to `None`, simplifying default string validation behavior.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_280

LANGUAGE: python
CODE:
```
from pydantic import BaseModel

class ModelWithConfig(BaseModel):
    class Config:
        # min_anystr_length and max_anystr_length are None by default
```

----------------------------------------

TITLE: Type Annotations and Mypy Testing
DESCRIPTION: Enhanced code quality by adding type annotations to all functions and performing comprehensive testing with mypy, improving static analysis and maintainability.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_272

LANGUAGE: python
CODE:
```
def process_data(data: dict) -> list:
    # ... implementation ...
```

----------------------------------------

TITLE: Pydantic v1.0b2 API Changes
DESCRIPTION: Details API and behavior changes introduced in Pydantic v1.0b2. This includes type checking adjustments for `StrictBool`, documentation build system migration, support for custom naming schemes in `GenericModel`, and renaming of a configuration parameter.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_256

LANGUAGE: APIDOC
CODE:
```
Pydantic API Changes (v1.0b2):

- Mark `StrictBool` typecheck as `bool` to allow default values without mypy errors.
- Add support for custom naming schemes for `GenericModel` subclasses.
- Rename `allow_population_by_alias` to `allow_population_by_field_name`; remove related warning.
```

----------------------------------------

TITLE: Support arbitrary types with custom __eq__
DESCRIPTION: Adds support for arbitrary types that define a custom `__eq__` method. Pydantic can now correctly compare and validate models containing such custom types.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_131

LANGUAGE: python
CODE:
```
support arbitrary types with custom `__eq__`, [#2483](https://github.com/pydantic/pydantic/pull/2483) by @PrettyWood
```

----------------------------------------

TITLE: Pydantic Dataclass Decorator
DESCRIPTION: The `pydantic.dataclasses.dataclass` decorator now supports built-in `dataclasses.dataclass`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_216

LANGUAGE: python
CODE:
```
import dataclasses
from pydantic.dataclasses import dataclass

# Using Pydantic's dataclass decorator which now wraps standard dataclasses
@dataclass
class MyData:
    name: str
    value: int

# This behaves like a standard dataclass but with Pydantic's validation features.
# instance = MyData(name='example', value=10)
# print(instance.name)

```

----------------------------------------

TITLE: Pydantic NamedTuple Validation
DESCRIPTION: Demonstrates Pydantic's validation for `typing.NamedTuple` and `collections.namedtuple`. It shows how Pydantic creates instances of the specified namedtuple class and validates fields, including type coercion and error handling for invalid inputs.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/standard_library_types.md#_snippet_10

LANGUAGE: python
CODE:
```
from typing import NamedTuple

from pydantic import BaseModel, ValidationError


class Point(NamedTuple):
    x: int
    y: int


class Model(BaseModel):
    p: Point


try:
    Model(p=('1.3', '2'))
except ValidationError as e:
    print(e)
```

----------------------------------------

TITLE: Custom Encoding for Dotenv Files
DESCRIPTION: Adds support for specifying custom encoding when loading configuration from `.env` files. This feature provides flexibility for projects using non-standard file encodings.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_223

LANGUAGE: python
CODE:
```
from pydantic import BaseSettings

class Settings(BaseSettings):
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8' # or 'latin-1', etc.

# This allows loading .env files with specified encodings.
```

----------------------------------------

TITLE: Pydantic Field Customization with Field()
DESCRIPTION: Demonstrates basic customization of Pydantic model fields using the Field() function for required fields and frozen attributes.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/fields.md#_snippet_0

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field


class Model(BaseModel):
    name: str = Field(frozen=True)
```

----------------------------------------

TITLE: Custom Validation with Annotated and AfterValidator
DESCRIPTION: Demonstrates creating a custom validator using Pydantic's Annotated type and the `__get_pydantic_core_schema__` method. This allows applying specific validation functions, like `str.lower`, to types within Pydantic models. It requires `pydantic` and `pydantic_core`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/types.md#_snippet_13

LANGUAGE: python
CODE:
```
from dataclasses import dataclass
from typing import Annotated, Any, Callable

from pydantic_core import CoreSchema, core_schema

from pydantic import BaseModel, GetCoreSchemaHandler


@dataclass(frozen=True)
class MyAfterValidator:
    func: Callable[[Any], Any]

    def __get_pydantic_core_schema__(
        self,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            self.func, handler(source_type)
        )


Username = Annotated[str, MyAfterValidator(str.lower)]


class Model(BaseModel):
    name: Username


assert Model(name='ABC').name == 'abc'
```

----------------------------------------

TITLE: Pydantic: Config Options for String Handling
DESCRIPTION: Details the `Config.anystr_lower` option and the `to_lower` kwarg for `constr` and `conbytes`. These enable automatic lowercasing of string and bytes types during validation.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_172

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field, constr

class ModelWithLowerString(BaseModel):
    class Config:
        anystr_lower = True

    name: str
    code: constr(to_lower=True)

# Example usage:
# model = ModelWithLowerString(name='TesT', code='VAL')
# print(model.name) # Output: 'test'
# print(model.code) # Output: 'val'
```

----------------------------------------

TITLE: Pydantic resolve_annotations Lenience
DESCRIPTION: Makes the `resolve_annotations` utility more lenient by allowing it to handle cases where modules might be missing. This improves robustness when resolving type annotations in complex or incomplete environments.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_144

LANGUAGE: APIDOC
CODE:
```
Pydantic resolve_annotations Lenience:

`resolve_annotations` is now more lenient, allowing for missing modules during annotation resolution.
```

----------------------------------------

TITLE: Pydantic Model with Custom Field Serializer
DESCRIPTION: Demonstrates defining a Pydantic `BaseModel` with a boolean field and a custom serializer using the `@field_serializer` decorator. This custom logic is then incorporated into the model's core schema for serialization.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/internals/architecture.md#_snippet_4

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field
from pydantic.functional_serializers import field_serializer


class Model(BaseModel):
    foo: bool = Field(strict=True)

    @field_serializer('foo', mode='plain')
    def serialize_foo(self, value: bool) -> int:
        # Example serialization logic
        return int(value) # Convert boolean to integer
```

----------------------------------------

TITLE: Pydantic Error Model with Nested Details
DESCRIPTION: Demonstrates creating a Pydantic model with nested details and serializing it using `model_dump()`. Shows how missing fields in nested models are handled during serialization.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_38

LANGUAGE: python
CODE:
```
from pydantic import BaseModel

class ErrorDetails(BaseModel):
    foo: str

class Error(BaseModel):
    message: str
    details: ErrorDetails

error = Error(
    message='We just had an error',
    details=ErrorDetails(foo='var'),
)
assert error.model_dump() == {
    'message': 'We just had an error',
    'details': {
        'foo': 'var',
    },
}
```

----------------------------------------

TITLE: Configuring JsonSchemaMode for Validation and Serialization
DESCRIPTION: Illustrates how to configure the JSON schema generation mode using the `mode` parameter in `model_json_schema` and `TypeAdapter.json_schema`. It shows the difference between 'validation' mode (default) and 'serialization' mode for a Decimal field.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/json_schema.md#_snippet_4

LANGUAGE: python
CODE:
```
from decimal import Decimal

from pydantic import BaseModel


class Model(BaseModel):
    a: Decimal = Decimal('12.34')


print(Model.model_json_schema(mode='validation'))
```

LANGUAGE: python
CODE:
```
from decimal import Decimal

from pydantic import BaseModel


class Model(BaseModel):
    a: Decimal = Decimal('12.34')


print(Model.model_json_schema(mode='serialization'))
```

----------------------------------------

TITLE: Pydantic Union Validation Errors Comparison
DESCRIPTION: Illustrates the verbosity of validation errors with standard Unions in Pydantic, especially with recursion, and highlights how discriminated unions simplify these messages by only showing errors for the matching discriminator case.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/unions.md#_snippet_8

LANGUAGE: python
CODE:
```
from typing import Annotated, Union

from pydantic import BaseModel, Discriminator, Tag, ValidationError


# Errors are quite verbose with a normal Union:
class Model(BaseModel):
    x: Union[str, 'Model']


try:
    Model.model_validate({'x': {'x': {'x': 1}}})
except ValidationError as e:
    print(e)
    """
    4 validation errors for Model
    x.str
      Input should be a valid string [type=string_type, input_value={'x': {'x': 1}}, input_type=dict]
    x.Model.x.str
      Input should be a valid string [type=string_type, input_value={'x': 1}, input_type=dict]
    x.Model.x.Model.x.str
      Input should be a valid string [type=string_type, input_value=1, input_type=int]
    x.Model.x.Model.x.Model
      Input should be a valid dictionary or instance of Model [type=model_type, input_value=1, input_type=int]
    """

try:
    Model.model_validate({'x': {'x': {'x': {}}}})
except ValidationError as e:
    print(e)
    """
    4 validation errors for Model
    x.str
      Input should be a valid string [type=string_type, input_value={'x': {'x': {}}}, input_type=dict]
    x.Model.x.str
      Input should be a valid string [type=string_type, input_value={'x': {}}, input_type=dict]
    x.Model.x.Model.x.str
      Input should be a valid string [type=string_type, input_value={}, input_type=dict]
    x.Model.x.Model.x.Model.x
      Field required [type=missing, input_value={}, input_type=dict]
    """

```

----------------------------------------

TITLE: Class Not Fully Defined: Using model_rebuild
DESCRIPTION: Shows the correct way to resolve 'class-not-fully-defined' errors for BaseModel subclasses by defining all types and then calling `.model_rebuild()` to update the model's internal structure.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/usage_errors.md#_snippet_2

LANGUAGE: python
CODE:
```
from typing import Optional

from pydantic import BaseModel


class Foo(BaseModel):
    a: Optional['Bar'] = None


class Bar(BaseModel):
    b: 'Foo'


Foo.model_rebuild()

foo = Foo(a={'b': {'a': None}})
```

----------------------------------------

TITLE: Pydantic ValidationError and ErrorDetails
DESCRIPTION: API documentation for Pydantic's ValidationError object and its associated ErrorDetails structure, outlining methods for error retrieval and properties for detailed error information.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/errors.md#_snippet_1

LANGUAGE: APIDOC
CODE:
```
PydanticValidationError:
  Methods:
    errors() -> list[ErrorDetails]
      - Returns a list of ErrorDetails objects representing all validation errors found.
    error_count() -> int
      - Returns the total number of validation errors.
    json() -> str
      - Returns a JSON string representation of the list of errors.
    str(e) -> str
      - Returns a human-readable string representation of the validation errors.

ErrorDetails:
  Properties:
    ctx: object | None
      - An optional object containing values required to render the error message.
    input: Any
      - The input data that caused the validation error.
    loc: list[str | int]
      - The location of the error within the data structure, as a list of keys or indices.
    msg: str
      - A human-readable explanation of the validation error.
    type: str
      - A computer-readable identifier for the type of validation error.
    url: str | None
      - A URL pointing to documentation for this specific error type.

Note:
  Validation code should raise ValueError or AssertionError, which Pydantic catches and converts into ValidationError.
```

----------------------------------------

TITLE: Pydantic Self-Referencing Models
DESCRIPTION: Illustrates how Pydantic supports models with self-referencing fields, using string annotations to resolve types within the model itself. This is useful for creating linked data structures.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/forward_annotations.md#_snippet_1

LANGUAGE: python
CODE:
```
from typing import Optional

from pydantic import BaseModel


class Foo(BaseModel):
    a: int = 123
    sibling: 'Optional[Foo]' = None


print(Foo())
#> a=123 sibling=None
print(Foo(sibling={'a': '321'}))
#> a=123 sibling=Foo(a=321, sibling=None)
```

----------------------------------------

TITLE: PostgresDsn Multi-Host Validation
DESCRIPTION: Enhances the `PostgresDsn` type to support validation of connection strings with multiple hosts, improving flexibility for PostgreSQL deployments.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_103

LANGUAGE: python
CODE:
```
from pydantic import BaseModel
from pydantic.networks import PostgresDsn

class DBConfigMultiHost(BaseModel):
    # Supports connection strings like: postgresql://user:pass@host1:port,host2:port/dbname
    db_connection: PostgresDsn

# Example:
# config = DBConfigMultiHost(db_connection='postgresql://user:pass@host1:5432,host2:5432/mydatabase')
# This allows Pydantic to parse and validate multi-host PostgreSQL URLs.
```

----------------------------------------

TITLE: Python Postponed Annotation Evaluation
DESCRIPTION: Illustrates the use of `from __future__ import annotations` to enable postponed evaluation of type hints in Python. This feature stringifies annotations by default, allowing them to be resolved later, as shown with Pydantic's `BaseModel`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/internals/resolving_annotations.md#_snippet_1

LANGUAGE: Python
CODE:
```
from __future__ import annotations

from pydantic import BaseModel


class Foo(BaseModel):
    f: MyType
    # Given the future import above, this is equivalent to:
    # f: 'MyType'


type MyType = int

print(Foo.__annotations__)
#> {'f': 'MyType'}
```

----------------------------------------

TITLE: Pydantic Validation by Alias and Name (`by_alias=True`, `by_name=True`)
DESCRIPTION: Shows how to enable validation using both aliases and original field names. Pydantic prioritizes aliases when both `by_alias` and `by_name` are `True`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/alias.md#_snippet_14

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field


class Model(BaseModel):
    my_field: str = Field(validation_alias='my_alias')


m = Model.model_validate(
    {'my_alias': 'foo'},
    by_alias=True,
    by_name=True
)
print(repr(m))
#> Model(my_field='foo')

m = Model.model_validate(
    {'my_field': 'foo'},
    by_alias=True,
    by_name=True
)
print(repr(m))
#> Model(my_field='foo')
```

----------------------------------------

TITLE: Mypy Plugin: Default/Default Factory Checks
DESCRIPTION: Adds checks within the Pydantic MyPy plugin for `default` and `default_factory` arguments, improving type safety and catching potential issues early.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_100

LANGUAGE: python
CODE:
```
# This enhancement to the MyPy plugin helps catch type errors related to default values
# in Pydantic models, making static analysis more effective.
```

----------------------------------------

TITLE: Pydantic: RedisDsn Protocol Support (rediss)
DESCRIPTION: Extends `RedisDsn` to support the `rediss` protocol (Redis over SSL) and allows URLs without a user part.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_182

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, RedisDsn

class RedisConfig(BaseModel):
    redis_url: RedisDsn

# Example usage:
# valid_url_ssl = RedisConfig(redis_url='rediss://user:password@host:6379/db')
# valid_url_no_user = RedisConfig(redis_url='rediss://localhost:6379')
# print(valid_url_ssl.redis_url)
# print(valid_url_no_user.redis_url)
```

----------------------------------------

TITLE: Default Factory Singleton Behavior
DESCRIPTION: Refines the behavior of `default_factory` to call the factory function only once when possible and to avoid setting a default value in the schema if the factory is used. This prevents unexpected side effects and ensures correct singleton instantiation.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_229

LANGUAGE: python
CODE:
```
from pydantic import BaseModel
from typing import List

class MyModel(BaseModel):
    items: List[int] = Field(default_factory=list)

# With the fix, if default_factory is a singleton or a factory that should only be called once,
# Pydantic handles it more predictably.
```

----------------------------------------

TITLE: Pydantic BaseModel Equality Changes
DESCRIPTION: Details the updated equality comparison rules for Pydantic BaseModel instances in V2. Models are now strictly compared based on type, field values, extra values, and private attributes, disallowing comparison with dictionaries.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_11

LANGUAGE: APIDOC
CODE:
```
Pydantic BaseModel Equality Rules (V2):

Models can only be equal to other `BaseModel` instances.
For two model instances to be equal, they must have the same:

1.  **Type**: Must be the same class or, for generic models, the same non-parametrized generic origin type.
2.  **Field Values**: All defined fields must have identical values.
3.  **Extra Values**: If `model_config['extra'] == 'allow'`, extra fields must match.
4.  **Private Attribute Values**: Models with different private attribute values are no longer considered equal.

Key Changes from V1:
- Models are no longer equal to dictionaries containing their data.
- Non-generic models of different types are never equal.
- Generic models with different origin types are never equal (e.g., `MyGenericModel[Any]` vs. `MyGenericModel[int]`).

Example Scenario:
```python
from pydantic import BaseModel

class UserV1(BaseModel):
    id: int

class UserV2(BaseModel):
    id: int

user1 = UserV2(id=1)
user2 = UserV2(id=1)
user3 = UserV1(id=1)
user_dict = {'id': 1}

# V2 Behavior:
print(user1 == user2)  # True
print(user1 == user3)  # False (different types)
print(user1 == user_dict) # False (cannot compare model to dict)
```
```

----------------------------------------

TITLE: Pydantic Callable Field Validation
DESCRIPTION: Shows how to define a Pydantic `BaseModel` with a `Callable` field and validates an instance with a lambda function. Notes that only callability is checked, not argument/return types.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/standard_library_types.md#_snippet_20

LANGUAGE: Python
CODE:
```
from typing import Callable

from pydantic import BaseModel


class Foo(BaseModel):
    callback: Callable[[int], int]


m = Foo(callback=lambda x: x)
print(m)
#> callback=<function <lambda> at 0x0123456789ab>

!!! warning
    Callable fields only perform a simple check that the argument is
    callable; no validation of arguments, their types, or the return
    type is performed.
```

----------------------------------------

TITLE: Pydantic Field Customization with Ellipsis
DESCRIPTION: Shows how to use the ellipsis (...) with Field() to explicitly mark a field as required, even when a value is assigned.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/fields.md#_snippet_1

LANGUAGE: python
CODE:
```
class Model(BaseModel):
    name: str = Field(..., frozen=True)
```

----------------------------------------

TITLE: Pydantic TypeAdapter for Union Validation
DESCRIPTION: Shows how to use Pydantic's TypeAdapter for validating data against a union type, offering an alternative to inheriting from BaseModel when only validation is needed.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/unions.md#_snippet_7

LANGUAGE: python
CODE:
```
from typing import Annotated, Union

from pydantic import BaseModel, Field, TypeAdapter


class BlackCat(BaseModel):
    pet_type: Literal['cat']
    color: Literal['black']
    black_name: str


class WhiteCat(BaseModel):
    pet_type: Literal['cat']
    color: Literal['white']
    white_name: str


Cat = Annotated[Union[BlackCat, WhiteCat], Field(discriminator='color')]


class Dog(BaseModel):
    pet_type: Literal['dog']
    name: str


Pet = Annotated[Union[Cat, Dog], Field(discriminator='pet_type')]


type_adapter = TypeAdapter(Pet)

pet = type_adapter.validate_python(
    {'pet_type': 'cat', 'color': 'black', 'black_name': 'felix'}
)
print(repr(pet))
#> BlackCat(pet_type='cat', color='black', black_name='felix')

```

----------------------------------------

TITLE: Field Inclusion and Exclusion via Serialization Parameters
DESCRIPTION: Demonstrates excluding and including specific fields during Pydantic model serialization using the `exclude` and `include` parameters in serialization methods like `model_dump()`. This allows for dynamic control over the serialized output.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/serialization.md#_snippet_20

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field, SecretStr


class User(BaseModel):
    id: int
    username: str
    password: SecretStr


class Transaction(BaseModel):
    id: str
    private_id: str = Field(exclude=True)
    user: User
    value: int

t = Transaction(
    id='1234567890',
    private_id='123',
    user=User(id=42, username='JohnDoe', password='hashedpassword'),
    value=9876543210,
)

# using a set:
print(t.model_dump(exclude={'user', 'value'}))

# using a dictionary:
print(t.model_dump(exclude={'user': {'username', 'password'}, 'value': True}))
```

----------------------------------------

TITLE: Pydantic Numeric Constraints JSON Schema
DESCRIPTION: Shows the JSON Schema generated from Pydantic models with numeric constraints, mapping Pydantic arguments to JSON Schema keywords.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/fields.md#_snippet_18

LANGUAGE: json
CODE:
```
{
  "title": "Foo",
  "type": "object",
  "properties": {
    "positive": {
      "title": "Positive",
      "type": "integer",
      "exclusiveMinimum": 0
    },
    "non_negative": {
      "title": "Non Negative",
      "type": "integer",
      "minimum": 0
    },
    "negative": {
      "title": "Negative",
      "type": "integer",
      "exclusiveMaximum": 0
    },
    "non_positive": {
      "title": "Non Positive",
      "type": "integer",
      "maximum": 0
    },
    "even": {
      "title": "Even",
      "type": "integer",
      "multipleOf": 2
    },
    "love_for_pydantic": {
      "title": "Love For Pydantic",
      "type": "number"
    }
  },
  "required": [
    "positive",
    "non_negative",
    "negative",
    "non_positive",
    "even",
    "love_for_pydantic"
  ]
}

```

----------------------------------------

TITLE: Pydantic Wrap Serializer - Decorator
DESCRIPTION: Demonstrates using the @field_serializer decorator with mode='wrap' to customize serialization. This approach includes a 'handler' parameter to integrate with Pydantic's default serialization process.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/serialization.md#_snippet_7

LANGUAGE: python
CODE:
```
from typing import Any

from pydantic import BaseModel, SerializerFunctionWrapHandler, field_serializer


class Model(BaseModel):
    number: int

    @field_serializer('number', mode='wrap')
    def ser_number(
        self, value: Any, handler: SerializerFunctionWrapHandler
    ) -> int:
        return handler(value) + 1


print(Model(number=4).model_dump())
#> {'number': 5}
```

----------------------------------------

TITLE: Pydantic root_validator Error Handling
DESCRIPTION: Raises a user-friendly `TypeError` when a `root_validator` fails to return a dictionary (e.g., returns `None`). This improves debugging by providing clearer error messages for validator misconfigurations.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_154

LANGUAGE: APIDOC
CODE:
```
Pydantic root_validator Error Handling:

Raises a user-friendly `TypeError` if a `root_validator` does not return a `dict` (e.g., returns `None`).
```

----------------------------------------

TITLE: Validating Partial JSON with Pydantic Models
DESCRIPTION: Combines `pydantic_core.from_json` with `BaseModel.model_validate` to parse and validate incomplete JSON data against a Pydantic model. This approach is useful for handling LLM outputs or other sources of potentially malformed JSON.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/json.md#_snippet_3

LANGUAGE: python
CODE:
```
from pydantic_core import from_json
from pydantic import BaseModel

class Dog(BaseModel):
    breed: str
    name: str
    friends: list

partial_dog_json = '{"breed": "lab", "name": "fluffy", "friends": ["buddy", "spot", "rufus"], "age'

dog = Dog.model_validate(from_json(partial_dog_json, allow_partial=True))
print(repr(dog))
#> Dog(breed='lab', name='fluffy', friends=['buddy', 'spot', 'rufus'])
```

----------------------------------------

TITLE: Pydantic V2 Float to Integer Conversion
DESCRIPTION: Details Pydantic V2's stricter float-to-integer conversion, allowing it only if the float has no fractional part. This prevents potential data loss that could occur in Pydantic V1 where any float was accepted for an `int` field.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_26

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ValidationError


class Model(BaseModel):
    x: int


# Valid conversion: float with zero decimal part
print(Model(x=10.0))
# Expected output: x=10

try:
    # Invalid conversion: float with non-zero decimal part
    Model(x=10.2)
except ValidationError as err:
    print(err)
    # Output will show validation error:
    # 1 validation error for Model
    # x
    #   Input should be a valid integer, got a number with a fractional part [type=int_from_float, input_value=10.2, input_type=float]
```

----------------------------------------

TITLE: Pydantic: Allow Extra Data
DESCRIPTION: Configures Pydantic models to allow extra data, storing it in the `__pydantic_extra__` attribute. This is controlled via the `extra` setting in `ConfigDict`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_11

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ConfigDict


class Model(BaseModel):
    x: int

    model_config = ConfigDict(extra='allow')


m = Model(x=1, y='a')  # (1)!
assert m.model_dump() == {'x': 1, 'y': 'a'}
assert m.__pydantic_extra__ == {'y': 'a'}
```

----------------------------------------

TITLE: AfterValidator Annotated Pattern (Check)
DESCRIPTION: Demonstrates using `AfterValidator` with the annotated pattern to check if an integer is even. Raises `ValueError` if the condition is not met. Requires `typing.Annotated` and `pydantic.AfterValidator`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/validators.md#_snippet_0

LANGUAGE: Python
CODE:
```
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ValidationError


def is_even(value: int) -> int:
    if value % 2 == 1:
        raise ValueError(f'{value} is not an even number')
    return value  # (1)!


class Model(BaseModel):
    number: Annotated[int, AfterValidator(is_even)]


try:
    Model(number=1)
except ValidationError as err:
    print(err)
    
```

----------------------------------------

TITLE: Pydantic Model `.dict()` and `.json()` Behavior
DESCRIPTION: Addresses an issue where internal `__root__` dictionaries were not properly squashed in the `.dict()` and `.json()` methods. This change ensures consistent output for models utilizing `__root__`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_240

LANGUAGE: python
CODE:
```
Squash internal `__root__` dicts in `.dict()` and `.json()`
Related Issue: [#1414](https://github.com/pydantic/pydantic/pull/1414)
```

----------------------------------------

TITLE: Define and Validate Pydantic Forward Reference
DESCRIPTION: This Python snippet defines 'MyInt' as an integer type, acting as a forward reference. It then rebuilds a Pydantic type adapter ('ta') to incorporate this new definition and validates an integer value against it, asserting the successful validation.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/type_adapter.md#_snippet_4

LANGUAGE: python
CODE:
```
# some time later, the forward reference is defined
MyInt = int

ta.rebuild()
assert ta.validate_python(1) == 1
```

----------------------------------------

TITLE: Decorator Pattern for Field Serialization
DESCRIPTION: Illustrates using the @field_serializer decorator to apply a serialization function to multiple fields. It covers options like applying to all fields ('*') and disabling field existence checks.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/serialization.md#_snippet_10

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, field_serializer


class Model(BaseModel):
    f1: str
    f2: str

    @field_serializer('f1', 'f2', mode='plain')
    def capitalize(self, value: str) -> str:
        return value.capitalize()
```

----------------------------------------

TITLE: Pydantic Wrap Validator for Datetime Field
DESCRIPTION: Illustrates customizing validation for a datetime field using a wrap validator in Pydantic V2. It shows how to handle a 'now' string input and ensure timezone awareness for naive datetimes by attaching UTC timezone.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/why.md#_snippet_6

LANGUAGE: python
CODE:
```
from datetime import datetime, timezone
from typing import Any

from pydantic_core.core_schema import ValidatorFunctionWrapHandler

from pydantic import BaseModel, field_validator


class Meeting(BaseModel):
    when: datetime

    @field_validator('when', mode='wrap')
    def when_now(
        cls, input_value: Any, handler: ValidatorFunctionWrapHandler
    ) -> datetime:
        if input_value == 'now':
            return datetime.now()
        when = handler(input_value)
        # in this specific application we know tz naive datetimes are in UTC
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return when


print(Meeting(when='2020-01-01T12:00+01:00'))
#> when=datetime.datetime(2020, 1, 1, 12, 0, tzinfo=TzInfo(3600))
print(Meeting(when='now'))
#> when=datetime.datetime(2032, 1, 2, 3, 4, 5, 6) # Example output, actual time will vary
print(Meeting(when='2020-01-01T12:00'))
#> when=datetime.datetime(2020, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)

```

----------------------------------------

TITLE: Class Not Fully Defined: ForwardRef
DESCRIPTION: Demonstrates the 'class-not-fully-defined' error when a type annotation uses ForwardRef to a class that is not yet defined. It shows how to catch the PydanticUserError.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/usage_errors.md#_snippet_0

LANGUAGE: python
CODE:
```
from typing import ForwardRef

from pydantic import BaseModel, PydanticUserError

UndefinedType = ForwardRef('UndefinedType')


class Foobar(BaseModel):
    a: UndefinedType


try:
    Foobar(a=1)
except PydanticUserError as exc_info:
    assert exc_info.code == 'class-not-fully-defined'
```

----------------------------------------

TITLE: Pydantic AliasChoices for Multiple Field Aliases
DESCRIPTION: Illustrates using AliasChoices to define multiple possible aliases for a single field, allowing flexibility during data validation. Shows validation with different alias combinations.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/alias.md#_snippet_1

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field, AliasChoices


class User(BaseModel):
    first_name: str = Field(validation_alias=AliasChoices('first_name', 'fname'))
    last_name: str = Field(validation_alias=AliasChoices('last_name', 'lname'))

user = User.model_validate({'fname': 'John', 'lname': 'Doe'})  # (1)!
print(user)
#> first_name='John' last_name='Doe'
user = User.model_validate({'first_name': 'John', 'lname': 'Doe'})  # (2)!
print(user)
#> first_name='John' last_name='Doe'
```

----------------------------------------

TITLE: Pydantic v0.30.1: Nested Class Initialization Fix
DESCRIPTION: Addresses an issue where nested classes inheriting from a parent and modifying `__init__` were not correctly processed, while still allowing `self` as a parameter.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_264

LANGUAGE: APIDOC
CODE:
```
Pydantic v0.30.1 Release Notes:

- Fix processing of nested classes that inherit and change `__init__`.
  - Ensures correct processing while allowing `self` as a parameter.
  - Related to issue [#644](https://github.com/pydantic/pydantic/pull/644).
  - Contributed by @lnaden and @dgasmith.
```

----------------------------------------

TITLE: Pydantic: Rebuild Model Schema with Forward Annotations
DESCRIPTION: Demonstrates how to use `model_rebuild()` to resolve forward references in model definitions, especially when a type is used before it's defined. This is crucial for complex or recursive structures.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_14

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, PydanticUserError


class Foo(BaseModel):
    x: 'Bar'  # (1)!


try:
    Foo.model_json_schema()
except PydanticUserError as e:
    print(e)
    """
    `Foo` is not fully defined; you should define `Bar`, then call `Foo.model_rebuild()`.

    For further information visit https://errors.pydantic.dev/2/u/class-not-fully-defined
    """


class Bar(BaseModel):
    pass


Foo.model_rebuild()
print(Foo.model_json_schema())
"""
{
    '$defs': {'Bar': {'properties': {}, 'title': 'Bar', 'type': 'object'}},
    'properties': {'x': {'$ref': '#/$defs/Bar'}},
    'required': ['x'],
    'title': 'Foo',
    'type': 'object',
}
"""
```

----------------------------------------

TITLE: Pydantic v1.0b2 Feature Additions
DESCRIPTION: Details new features introduced in Pydantic v1.0b2. This includes support for custom naming schemes for `GenericModel` subclasses and improvements to type checking for `StrictBool`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_260

LANGUAGE: APIDOC
CODE:
```
Pydantic Feature Additions (v1.0b2):

- Add support for custom naming schemes for `GenericModel` subclasses.
- Mark `StrictBool` typecheck as `bool` to allow for default values without mypy errors.
```

----------------------------------------

TITLE: Validate TOML Data with Pydantic in Python
DESCRIPTION: Validates data from a TOML file using Pydantic. It uses the `tomllib` module (standard library in Python 3.11+) to load TOML data and then validates it against a Pydantic model. Requires the `pydantic` library.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/examples/files.md#_snippet_5

LANGUAGE: python
CODE:
```
import tomllib

from pydantic import BaseModel, EmailStr, PositiveInt


class Person(BaseModel):
    name: str
    age: PositiveInt
    email: EmailStr


with open('person.toml', 'rb') as f:
    data = tomllib.load(f)

person = Person.model_validate(data)
print(person)
#> name='John Doe' age=30 email='john@example.com'
```

----------------------------------------

TITLE: Validate Incomplete Python Object with Partial Validation
DESCRIPTION: Shows that partial validation also works with Python objects, mirroring the behavior of JSON validation. Incomplete or invalid Python data is handled gracefully, omitting problematic parts.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/experimental.md#_snippet_9

LANGUAGE: python
CODE:
```
from typing import Annotated

from annotated_types import MinLen
from typing_extensions import NotRequired, TypedDict

from pydantic import TypeAdapter


class Foobar(TypedDict):
    a: int
    b: NotRequired[float]
    c: NotRequired[Annotated[str, MinLen(5)]]


ta = TypeAdapter(list[Foobar])

v = ta.validate_python([{'a': 1}], experimental_allow_partial=True)
print(v)
#> [{'a': 1}]
```

----------------------------------------

TITLE: Python: Data Loss with Unparametrized Generics
DESCRIPTION: Illustrates potential data loss when using unparametrized generic models with specific subtypes, contrasting explicit parametrization. Shows how validation against an upper bound without explicit parametrization can lose specific type information.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_36

LANGUAGE: python
CODE:
```
from typing import Generic, TypeVar

from pydantic import BaseModel

ItemT = TypeVar('ItemT', bound='ItemBase')


class ItemBase(BaseModel):
    pass


class IntItem(ItemBase):
    value: int


class ItemHolder(BaseModel, Generic[ItemT]):
    item: ItemT


loaded_data = {'item': {'value': 1}}


print(ItemHolder(**loaded_data))

print(ItemHolder[IntItem](**loaded_data))
```

----------------------------------------

TITLE: Strict Mode for Dataclasses and TypedDict
DESCRIPTION: Explains how to apply strict mode to Pydantic dataclasses using the `config` argument in the `@dataclass` decorator. For vanilla dataclasses or `TypedDict` subclasses, strict mode can be enabled by annotating fields with `pydantic.types.Strict` or by setting the `__pydantic_config__` attribute.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/strict_mode.md#_snippet_16

LANGUAGE: python
CODE:
```
from typing_extensions import TypedDict

from pydantic import ConfigDict, TypeAdapter, ValidationError


class Inner(TypedDict):
    y: int


Inner.__pydantic_config__ = ConfigDict(strict=True)


class Outer(TypedDict):
    x: int
    inner: Inner


adapter = TypeAdapter(Outer)
print(adapter.validate_python({'x': '1', 'inner': {'y': 2}}))

try:
    adapter.validate_python({'x': '1', 'inner': {'y': '2'}})
except ValidationError as exc:
    print(exc)
    
```

----------------------------------------

TITLE: Model Rebuild After Defining Forward Reference
DESCRIPTION: Shows how to resolve a forward reference by defining the type (e.g., `type MyType = int`) and then calling `Foo.model_rebuild()`. After rebuilding, `__pydantic_core_schema__` is correctly generated, reflecting the resolved type.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/internals/resolving_annotations.md#_snippet_8

LANGUAGE: python
CODE:
```
type MyType = int

Foo.model_rebuild()
Foo.__pydantic_core_schema__
```

----------------------------------------

TITLE: Marking Fields as Deprecated
DESCRIPTION: Shows how to use the `deprecated` parameter with a string message to mark Pydantic fields as deprecated. This generates a runtime deprecation warning upon access and sets the `deprecated` keyword in the JSON schema.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/fields.md#_snippet_30

LANGUAGE: python
CODE:
```
from typing import Annotated

from pydantic import BaseModel, Field


class Model(BaseModel):
    deprecated_field: Annotated[int, Field(deprecated='This is deprecated')]


print(Model.model_json_schema()['properties']['deprecated_field'])
#> {'deprecated': True, 'title': 'Deprecated Field', 'type': 'integer'}
```

----------------------------------------

TITLE: Pydantic: Validate arguments with validate_arguments
DESCRIPTION: Shows how to use the `validate_arguments` decorator to automatically validate function parameters. It also highlights the addition of a `validate` method for pre-call validation.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_170

LANGUAGE: python
CODE:
```
from pydantic import validate_arguments

@validate_arguments
def process_data(data: dict, count: int = 1):
    """Processes data with validated arguments."""
    print(f"Processing {count} items.")
    # ... processing logic ...

# Example of using the bound validate method:
# validator = process_data.validate(data={'key': 'value'}, count=5)
# validator.execute() # This would call the actual function
```

----------------------------------------

TITLE: Pydantic v1.0 Feature Additions
DESCRIPTION: Details new features and enhancements added in Pydantic v1.0. This includes improved handling of `**kwargs` in metaclasses, better `Field` constraint support for complex types, and enhanced `BaseSettings` merging capabilities.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_259

LANGUAGE: APIDOC
CODE:
```
Pydantic Feature Additions (v1.0):

- Add `**kwargs` to `pydantic.main.ModelMetaclass.__new__` for `__init_subclass__`.
- Improve use of `Field` constraints on complex types, support `Tuple[X, ...]`, `Sequence`, `FrozenSet` in schema.
- For `BaseSettings`, merge environment variables and in-code values recursively as long as they create a valid object when merged.
```

----------------------------------------

TITLE: Pydantic ConfigDict ser_json_inf_nan
DESCRIPTION: Introduces a new configuration option `ser_json_inf_nan` for `ConfigDict` in Pydantic. This setting controls how infinity and NaN values are serialized to JSON.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_51

LANGUAGE: python
CODE:
```
Add ConfigDict.ser_json_inf_nan by @davidhewitt in [#8159](https://github.com/pydantic/pydantic/pull/8159)
```

----------------------------------------

TITLE: Pydantic Parsing Helper Functions
DESCRIPTION: Introduced in v0.4.0, these functions provide convenient ways to parse data into Pydantic models from different sources. They handle data conversion and validation according to the model's schema.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_307

LANGUAGE: APIDOC
CODE:
```
parse_obj(obj: Any)
  Parses data from a Python object (e.g., dict) into a Pydantic model.
  Parameters:
    obj: The Python object containing the data to parse.
  Returns: An instance of the Pydantic model.

parse_raw(b: bytes | str, content_type: str | None = None, encoding: str | None = None)
  Parses data from raw bytes or string, inferring content type if not provided.
  Parameters:
    b: The raw bytes or string data.
    content_type: The content type of the data (e.g., 'application/json').
    encoding: The encoding of the data.
  Returns: An instance of the Pydantic model.

parse_file(path: str | Path, content_type: str | None = None, encoding: str | None = None)
  Parses data from a file path, reading the file content and validating it.
  Parameters:
    path: The file path to read data from.
    content_type: The content type of the file.
    encoding: The encoding of the file.
  Returns: An instance of the Pydantic model.
```

----------------------------------------

TITLE: Support kw_only in Dataclasses
DESCRIPTION: Adds support for the `kw_only` (keyword-only) argument in Pydantic's integration with Python dataclasses, allowing for more explicit argument passing.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_96

LANGUAGE: python
CODE:
```
from dataclasses import dataclass
from pydantic import dataclasses

@dataclasses.dataclass(kw_only=True)
class KeywordOnlyData:
    name: str
    value: int

# Usage:
# instance = KeywordOnlyData(name='test', value=10) # Valid
# instance_invalid = KeywordOnlyData('test', 10) # Raises TypeError
```

----------------------------------------

TITLE: Custom Environment Variable Parsing
DESCRIPTION: Introduces the ability to customize the parsing of environment variables through the `parse_env_var` setting within the `Config` class. This allows for more flexible handling of environment variable inputs.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_70

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field

class Settings(BaseModel):
    class Config:
        @classmethod
        def parse_env_var(cls, field_name: str, value: str) -> object:
            # Custom parsing logic here
            if field_name == 'my_custom_field':
                return int(value) * 2
            return value

    my_custom_field: int

```

----------------------------------------

TITLE: Fix Schema from ConstrainedStr with regex
DESCRIPTION: Fixes an issue where creating a schema from a model using `ConstrainedStr` with a regex pattern as a dictionary key would fail. This ensures correct schema generation for such cases.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_56

LANGUAGE: python
CODE:
```
# Fix creating schema from model using `ConstrainedStr` with `regex` as dict key, [#5223](https://github.com/pydantic/pydantic/pull/5223) by @matejetz
```

----------------------------------------

TITLE: Pydantic: Support for Plain typing.Tuple
DESCRIPTION: Enables the use of `typing.Tuple` without specifying element types, which is useful for tuples of mixed types where the exact sequence is not critical or is handled dynamically.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_173

LANGUAGE: python
CODE:
```
from typing import Tuple
from pydantic import BaseModel

class ModelWithTuple(BaseModel):
    data: Tuple

# Example usage:
# model = ModelWithTuple(data=(1, 'hello', True))
# print(model.data) # Output: (1, 'hello', True)
```

----------------------------------------

TITLE: MyPy Plugin: Prevent __init__ Override
DESCRIPTION: Fixes the Pydantic MyPy plugin to avoid overriding pre-existing `__init__` methods in models, ensuring custom initializers are preserved.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_92

LANGUAGE: python
CODE:
```
# This is a fix for the Pydantic MyPy plugin.
# It ensures that if a user defines a custom __init__ method,
# the plugin does not interfere with it.
```

----------------------------------------

TITLE: Pydantic: Both validate_by_alias and validate_by_name False
DESCRIPTION: Explains the Pydantic error raised when both `validate_by_alias` and `validate_by_name` are set to `False` in `ConfigDict`. This configuration is prohibited as it prevents attribute population.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/usage_errors.md#_snippet_56

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, ConfigDict, Field, PydanticUserError

try:

    class Model(BaseModel):
        a: int = Field(alias='A')

        model_config = ConfigDict(
            validate_by_alias=False, validate_by_name=False
        )

except PydanticUserError as exc_info:
    assert exc_info.code == 'validate-by-alias-and-name-false'
```

----------------------------------------

TITLE: Pydantic BaseModel.construct() Field Order
DESCRIPTION: Ensures that the order of fields is preserved when creating models using `BaseModel.construct()`. This is important for maintaining predictable field ordering, especially in scenarios where order matters.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_148

LANGUAGE: APIDOC
CODE:
```
Pydantic BaseModel.construct() Field Order:

Preserves the order of fields when using `BaseModel.construct()`.
```

----------------------------------------

TITLE: Pydantic Validation Concept
DESCRIPTION: Pydantic uses the term 'validation' to describe the process of instantiating a model or type that adheres to specified types and constraints. This process guarantees the types and constraints of the output, not the input data, and may involve parsing, coercion, and copying data without mutating the original input.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_1

LANGUAGE: APIDOC
CODE:
```
Validation — a *deliberate* misnomer

TL;DR
We use the term "validation" to refer to the process of instantiating a model (or other type) that adheres to specified types and constraints. This task, which Pydantic is well known for, is most widely recognized as "validation" in colloquial terms, even though in other contexts the term "validation" may be more restrictive.

--- 

The long version
The potential confusion around the term "validation" arises from the fact that, strictly speaking, Pydantic's primary focus doesn't align precisely with the dictionary definition of "validation":

> validation
> _noun_
> the action of checking or proving the validity or accuracy of something.

In Pydantic, the term "validation" refers to the process of instantiating a model (or other type) that adheres to specified types and constraints. Pydantic guarantees the types and constraints of the output, not the input data. This distinction becomes apparent when considering that Pydantic's `ValidationError` is raised when data cannot be successfully parsed into a model instance.

While this distinction may initially seem subtle, it holds practical significance. In some cases, "validation" goes beyond just model creation, and can include the copying and coercion of data. This can involve copying arguments passed to the constructor in order to perform coercion to a new type without mutating the original input data. For a more in-depth understanding of the implications for your usage, refer to the [Data Conversion](#data-conversion) and [Attribute Copies](#attribute-copies) sections below.

In essence, Pydantic's primary goal is to assure that the resulting structure post-processing (termed "validation") precisely conforms to the applied type hints. Given the widespread adoption of "validation" as the colloquial term for this process, we will consistently use it in our documentation.

While the terms "parse" and "validation" were previously used interchangeably, moving forward, we aim to exclusively employ "validate", with "parse" reserved specifically for discussions related to [JSON parsing](../concepts/json.md).
```

----------------------------------------

TITLE: Pydantic: Custom Datetime TZ Constraint Validator
DESCRIPTION: Demonstrates creating a Pydantic validator for datetime objects using Annotated metadata and __get_pydantic_core_schema__. This validator enforces a specific timezone constraint by wrapping the default validation logic. It requires pydantic, pytz, and pydantic-core.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/examples/custom_validators.md#_snippet_0

LANGUAGE: python
CODE:
```
import datetime as dt
from dataclasses import dataclass
from pprint import pprint
from typing import Annotated, Any, Callable, Optional

import pytz
from pydantic_core import CoreSchema, core_schema

from pydantic import (
    GetCoreSchemaHandler,
    PydanticUserError,
    TypeAdapter,
    ValidationError,
)


@dataclass(frozen=True)
class MyDatetimeValidator:
    tz_constraint: Optional[str] = None

    def tz_constraint_validator(
        self,
        value: dt.datetime,
        handler: Callable,  # (1)!
    ):
        """Validate tz_constraint and tz_info."""
        # handle naive datetimes
        if self.tz_constraint is None:
            assert (
                value.tzinfo is None
            ), 'tz_constraint is None, but provided value is tz-aware.'
            return handler(value)

        # validate tz_constraint and tz-aware tzinfo
        if self.tz_constraint not in pytz.all_timezones:
            raise PydanticUserError(
                f'Invalid tz_constraint: {self.tz_constraint}',
                code='unevaluable-type-annotation',
            )
        result = handler(value)  # (2)!
        assert self.tz_constraint == str(
            result.tzinfo
        ), f'Invalid tzinfo: {str(result.tzinfo)}, expected: {self.tz_constraint}'

        return result

    def __get_pydantic_core_schema__(
        self,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        return core_schema.no_info_wrap_validator_function(
            self.tz_constraint_validator,
            handler(source_type),
        )


LA = 'America/Los_Angeles'
ta = TypeAdapter(Annotated[dt.datetime, MyDatetimeValidator(LA)])
print(
    ta.validate_python(dt.datetime(2023, 1, 1, 0, 0, tzinfo=pytz.timezone(LA)))
)
#> 2023-01-01 00:00:00-07:53

LONDON = 'Europe/London'
try:
    ta.validate_python(
        dt.datetime(2023, 1, 1, 0, 0, tzinfo=pytz.timezone(LONDON))
    )
except ValidationError as ve:
    pprint(ve.errors(), width=100)
    """
    [{'ctx': {'error': AssertionError('Invalid tzinfo: Europe/London, expected: America/Los_Angeles')},
    'input': datetime.datetime(2023, 1, 1, 0, 0, tzinfo=<DstTzInfo 'Europe/London' LMT-1 day, 23:59:00 STD>),
    'loc': (),
    'msg': 'Assertion failed, Invalid tzinfo: Europe/London, expected: America/Los_Angeles',
    'type': 'assertion_error',
    'url': 'https://errors.pydantic.dev/2.8/v/assertion_error'}]
    """

```

----------------------------------------

TITLE: validate_call: Keyword-Only Parameters
DESCRIPTION: Illustrates the use of `validate_call` with keyword-only parameters, which must be explicitly named when called. This ensures clarity and prevents accidental positional argument passing.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/validation_decorator.md#_snippet_3

LANGUAGE: python
CODE:
```
from pydantic import validate_call


@validate_call
def kw_only(*, a: int, b: int = 2) -> str:
    return f'a={a} b={b}'


print(kw_only(a=1))
#> a=1 b=2
print(kw_only(a=1, b=3))
#> a=1 b=3
```

----------------------------------------

TITLE: Define Generic Pydantic Model with Custom Schema
DESCRIPTION: Demonstrates defining a generic class (`Owner`) and integrating it with Pydantic's core schema generation. This allows custom validation logic for generic type arguments within Pydantic models.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/types.md#_snippet_18

LANGUAGE: python
CODE:
```
from dataclasses import dataclass
from typing import Generic, TypeVar, Any

from pydantic_core import core_schema
from pydantic import GetCoreSchemaHandler, ValidationError
from pydantic.generics import GenericModel
from pydantic import BaseModel

ItemType = TypeVar('ItemType')

# This is not a pydantic model, it's an arbitrary generic class
@dataclass
class Owner(Generic[ItemType]):
    name: str
    item: ItemType

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        origin = get_origin(source_type)
        if origin is None:  # used as `x: Owner` without params
            origin = source_type
            item_tp = Any
        else:
            item_tp = get_args(source_type)[0]
        
        item_schema = handler.generate_schema(item_tp)

        def val_item(v: Owner[Any], handler: core_schema.ValidatorFunctionWrapHandler) -> Owner[Any]:
            v.item = handler(v.item)
            return v

        python_schema = core_schema.chain_schema(
            [
                core_schema.is_instance_schema(cls),
                core_schema.no_info_wrap_validator_function(val_item, item_schema),
            ]
        )

        return core_schema.json_or_python_schema(
            json_schema=core_schema.chain_schema(
                [
                    core_schema.typed_dict_schema(
                        {
                            'name': core_schema.typed_dict_field(
                                core_schema.str_schema()
                            ),
                            'item': core_schema.typed_dict_field(item_schema),
                        }
                    ),
                    core_schema.no_info_before_validator_function(
                        lambda data: Owner(
                            name=data['name'], item=data['item']
                        ),
                        python_schema,
                    ),
                ]
            ),
            python_schema=python_schema,
        )

# Helper functions for __get_pydantic_core_schema__
def get_origin(tp: Any) -> Any:
    return getattr(tp, '__origin__', None)

def get_args(tp: Any) -> tuple[Any, ...]:
    return getattr(tp, '__args__', ())

```

----------------------------------------

TITLE: Core Schema with Custom Serialization
DESCRIPTION: Shows a core schema definition that includes custom serialization logic for a field. The `serialization` key specifies a function-based serializer, indicating how the Python value should be transformed during the serialization process.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/internals/architecture.md#_snippet_3

LANGUAGE: python
CODE:
```
{
    'type': 'function-plain',
    'function': '<function Model.serialize_foo at 0x111>',
    'is_field_serializer': True,
    'info_arg': False,
    'return_schema': {'type': 'int'},
}
```

----------------------------------------

TITLE: FileUrl type and host_required parameter
DESCRIPTION: Introduces a new `FileUrl` type that conforms to RFC 8089 for file URLs. It also adds the `host_required` parameter, defaulting to `True` for `AnyUrl` and `False` for `RedisDsn` and `FileUrl`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_122

LANGUAGE: python
CODE:
```
Create `FileUrl` type that allows URLs that conform to [RFC 8089](https://tools.ietf.org/html/rfc8089#section-2).
  Add `host_required` parameter, which is `True` by default (`AnyUrl` and subclasses), `False` in `RedisDsn`, `FileUrl`, [#1983](https://github.com/pydantic/pydantic/pull/1983) by @vgerak
```

----------------------------------------

TITLE: Validate JSON Lines Data with Pydantic
DESCRIPTION: Demonstrates validating data from a JSON Lines (`.jsonl`) file, where each line is a separate JSON object. It reads the file line by line and validates each JSON object against a Pydantic model.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/examples/files.md#_snippet_3

LANGUAGE: python
CODE:
```
import pathlib

from pydantic import BaseModel, EmailStr, PositiveInt


class Person(BaseModel):
    name: str
    age: PositiveInt
    email: EmailStr


json_lines = pathlib.Path('people.jsonl').read_text().splitlines()
people = [Person.model_validate_json(line) for line in json_lines]
print(people)
#> [Person(name='John Doe', age=30, email='john@example.com'), Person(name='Jane Doe', age=25, email='jane@example.com')]
```

----------------------------------------

TITLE: Pydantic Runtime Serialization Methods API
DESCRIPTION: API documentation for Pydantic methods that support runtime alias control during serialization. These methods allow specifying whether to serialize using aliases via the `by_alias` flag.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/alias.md#_snippet_18

LANGUAGE: APIDOC
CODE:
```
BaseModel.model_dump(*, by_alias: bool = False, ...)
BaseModel.model_dump_json(*, by_alias: bool = False, ...)
TypeAdapter.dump_python(obj: Any, *, by_alias: bool = False, ...)
TypeAdapter.dump_json(obj: Any, *, by_alias: bool = False, ...)

Parameters:
  by_alias: bool (default: False) - If True, serialize using field aliases.
```

----------------------------------------

TITLE: Performance: Improve Model __setattr__
DESCRIPTION: Enhances the performance of the `__setattr__` method on Pydantic models by implementing caching for setter functions, leading to faster attribute assignments.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_32

LANGUAGE: python
CODE:
```
class MyModel:
    def __setattr__(self, name, value):
        # Optimized setter logic with caching
        super().__setattr__(name, value)

# This speeds up operations involving attribute modification on model instances.
```

----------------------------------------

TITLE: Annotated Pattern for Field Serialization
DESCRIPTION: Demonstrates using Annotated with PlainSerializer to apply custom serialization logic to fields. This pattern allows for reusable serializers defined directly within type annotations.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/serialization.md#_snippet_9

LANGUAGE: python
CODE:
```
from typing import Annotated

from pydantic import BaseModel, Field, PlainSerializer

DoubleNumber = Annotated[int, PlainSerializer(lambda v: v * 2)]


class Model1(BaseModel):
    my_number: DoubleNumber


class Model2(BaseModel):
    other_number: Annotated[DoubleNumber, Field(description='My other number')]


class Model3(BaseModel):
    list_of_even_numbers: list[DoubleNumber]  # (1)!
```

----------------------------------------

TITLE: Pydantic V2 Union Type Preservation
DESCRIPTION: Demonstrates how Pydantic V2 unions preserve the input type when possible, even if it's not the first type in the union. It contrasts this with Pydantic V1's behavior and mentions the `union_mode` setting to revert to V1's left-to-right validation.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_23

LANGUAGE: python
CODE:
```
from typing import Union

from pydantic import BaseModel


class Model(BaseModel):
    x: Union[int, str]


print(Model(x='1'))
# Expected output in Pydantic V2: x='1'
# Note: In Pydantic V1, this would output x=1.
```

----------------------------------------

TITLE: Custom Type: CompressedString with __get_pydantic_core_schema__
DESCRIPTION: Demonstrates creating a custom type, CompressedString, that overrides Pydantic's schema generation using __get_pydantic_core_schema__. It includes custom validation and serialization logic for a compressed string format.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/json_schema.md#_snippet_14

LANGUAGE: python
CODE:
```
from dataclasses import dataclass
from typing import Any

from pydantic_core import core_schema

from pydantic import BaseModel, GetCoreSchemaHandler


@dataclass
class CompressedString:
    dictionary: dict[int, str]
    text: list[int]

    def build(self) -> str:
        return ' '.join([self.dictionary[key] for key in self.text])

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: type[Any], handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        assert source is CompressedString
        return core_schema.no_info_after_validator_function(
            cls._validate,
            core_schema.str_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize,
                info_arg=False,
                return_schema=core_schema.str_schema(),
            ),
        )

    @staticmethod
    def _validate(value: str) -> 'CompressedString':
        inverse_dictionary: dict[str, int] = {}
        text: list[int] = []
        for word in value.split(' '):
            if word not in inverse_dictionary:
                inverse_dictionary[word] = len(inverse_dictionary)
            text.append(inverse_dictionary[word])
        return CompressedString(
            {v: k for k, v in inverse_dictionary.items()}, text
        )

    @staticmethod
    def _serialize(value: 'CompressedString') -> str:
        return value.build()


class MyModel(BaseModel):
    value: CompressedString


# Example Usage:
# print(MyModel.model_json_schema())
# print(MyModel(value='fox fox fox dog fox'))
# print(MyModel(value='fox fox fox dog fox').model_dump(mode='json'))

```

----------------------------------------

TITLE: Pydantic Custom Root Type from_orm() Support
DESCRIPTION: Adds support for custom root types (aka `__root__`) when using `from_orm()`. This allows Pydantic models to be created from ORM objects where the root data structure is custom.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_163

LANGUAGE: APIDOC
CODE:
```
Pydantic Custom Root Type from_orm() Support:

Supports custom root type (`__root__`) with `from_orm()`.
```

----------------------------------------

TITLE: Pydantic PlainValidator with Decorator
DESCRIPTION: Presents the decorator-based approach for `PlainValidator`. This method allows modifying input values directly and bypassing Pydantic's internal type validation, as demonstrated by doubling an integer or accepting a string input.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/validators.md#_snippet_7

LANGUAGE: python
CODE:
```
from typing import Any

from pydantic import BaseModel, field_validator


class Model(BaseModel):
    number: int

    @field_validator('number', mode='plain')
    @classmethod
    def val_number(cls, value: Any) -> Any:
        if isinstance(value, int):
            return value * 2
        else:
            return value


print(Model(number=4))
#> number=8
print(Model(number='invalid'))  # (1)!
#> number='invalid'

```

----------------------------------------

TITLE: Pydantic Python Version Requirement
DESCRIPTION: Updates the `python_requires` metadata to specify a minimum requirement of Python 3.6.1. This ensures compatibility with the features and syntax used in the library.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_157

LANGUAGE: APIDOC
CODE:
```
Pydantic Python Version Requirement:

`python_requires` metadata updated to require Python >= 3.6.1.
```

----------------------------------------

TITLE: Pydantic TypedDict Validation
DESCRIPTION: Demonstrates validating Python dictionaries against Pydantic `TypedDict` definitions, including cases with optional fields (`total=False`) and error handling for invalid types or extra fields.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/api/standard_library_types.md#_snippet_19

LANGUAGE: Python
CODE:
```
from typing import TypedDict, Optional
from pydantic import ConfigDict, TypeAdapter, ValidationError


# `total=False` means keys are non-required
class UserIdentity(TypedDict, total=False):
    name: Optional[str]
    surname: str


class User(TypedDict):
    __pydantic_config__ = ConfigDict(extra='forbid')

    identity: UserIdentity
    age: int


ta = TypeAdapter(User)

print(
    ta.validate_python(
        {'identity': {'name': 'Smith', 'surname': 'John'}, 'age': 37}
    )
)
#> {'identity': {'name': 'Smith', 'surname': 'John'}, 'age': 37}

print(
    ta.validate_python(
        {'identity': {'name': None, 'surname': 'John'}, 'age': 37}
    )
)
#> {'identity': {'name': None, 'surname': 'John'}, 'age': 37}

print(ta.validate_python({'identity': {}, 'age': 37}))
#> {'identity': {}, 'age': 37}


try:
    ta.validate_python(
        {'identity': {'name': ['Smith'], 'surname': 'John'}, 'age': 24}
    )
except ValidationError as e:
    print(e)
    """
    1 validation error for User
    identity.name
      Input should be a valid string [type=string_type, input_value=['Smith'], input_type=list]
    """

try:
    ta.validate_python(
        {
            'identity': {'name': 'Smith', 'surname': 'John'},
            'age': '37',
            'email': 'john.smith@me.com',
        }
    )
except ValidationError as e:
    print(e)
    """
    1 validation error for User
    email
      Extra inputs are not permitted [type=extra_forbidden, input_value='john.smith@me.com', input_type=str]
    """

```

----------------------------------------

TITLE: Performance: Optimize _typing_extra Module
DESCRIPTION: Refactors and optimizes the `_typing_extra` module, leading to performance gains in handling various typing constructs.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_34

LANGUAGE: python
CODE:
```
# Internal module optimization for enhanced performance.
```

----------------------------------------

TITLE: Pydantic AliasChoices with AliasPath
DESCRIPTION: Combines AliasChoices and AliasPath to provide flexible validation paths for fields, allowing either direct aliases or nested paths. Demonstrates validation with mixed alias strategies.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/alias.md#_snippet_2

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field, AliasPath, AliasChoices


class User(BaseModel):
    first_name: str = Field(validation_alias=AliasChoices('first_name', AliasPath('names', 0)))
    last_name: str = Field(validation_alias=AliasChoices('last_name', AliasPath('names', 1)))


user = User.model_validate({'first_name': 'John', 'last_name': 'Doe'})
print(user)
#> first_name='John' last_name='Doe'
user = User.model_validate({'names': ['John', 'Doe']})
print(user)
#> first_name='John' last_name='Doe'
user = User.model_validate({'names': ['John'], 'last_name': 'Doe'})
print(user)
#> first_name='John' last_name='Doe'
```

----------------------------------------

TITLE: Schema Generation for Dict, List, Tuple, Set
DESCRIPTION: Improved schema generation for fields annotated with generic collection types like `dict`, `list`, `tuple`, and `set`, ensuring correct schema output.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_285

LANGUAGE: python
CODE:
```
from pydantic import BaseModel
from typing import List, Dict, Tuple, Set

class CollectionModel(BaseModel):
    items_list: List[int]
    data_dict: Dict[str, float]
    coordinates: Tuple[int, int]
    unique_ids: Set[str]
```

----------------------------------------

TITLE: Send Validated Data to RabbitMQ Queue (Python)
DESCRIPTION: This sender script utilizes Pydantic to define a data structure for users and serializes instances of this model into JSON. The JSON data is then published as messages to a specified RabbitMQ queue using the pika library.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/examples/queues.md#_snippet_1

LANGUAGE: python
CODE:
```
import pika

from pydantic import BaseModel, EmailStr


class User(BaseModel):
    id: int
    name: str
    email: EmailStr


connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
QUEUE_NAME = 'user_queue'
channel.queue_declare(queue=QUEUE_NAME)


def push_to_queue(user_data: User) -> None:
    serialized_data = user_data.model_dump_json()
    channel.basic_publish(
        exchange='',
        routing_key=QUEUE_NAME,
        body=serialized_data,
    )
    print(f'Added to queue: {serialized_data}')


user1 = User(id=1, name='John Doe', email='john@example.com')
user2 = User(id=2, name='Jane Doe', email='jane@example.com')

push_to_queue(user1)
# > Added to queue: {"id":1,"name":"John Doe","email":"john@example.com"}

push_to_queue(user2)
# > Added to queue: {"id":2,"name":"Jane Doe","email":"jane@example.com"}

connection.close()
```

----------------------------------------

TITLE: NameEmail Equality Comparison
DESCRIPTION: Implements the `__eq__` method for `NameEmail` (likely a custom Pydantic type or model), allowing duplicate instances to be evaluated as equal. This is important for comparisons and set operations involving `NameEmail` objects.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_230

LANGUAGE: python
CODE:
```
from pydantic import BaseModel

class NameEmail(BaseModel):
    name: str
    email: str

    # The __eq__ method is now implemented to compare NameEmail instances
    # def __eq__(self, other):
    #     if not isinstance(other, NameEmail):
    #         return NotImplemented
    #     return self.name == other.name and self.email == other.email

# Example:
# email1 = NameEmail(name='Alice', email='alice@example.com')
# email2 = NameEmail(name='Alice', email='alice@example.com')
# print(email1 == email2) # Should print True
```

----------------------------------------

TITLE: Unpack Used Without TypedDict
DESCRIPTION: Shows the PydanticUserError raised when `typing.Unpack` is used with a type hint that is not a `typing.TypedDict` for variadic keyword parameters. This ensures correct usage with PEP 692.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/errors/usage_errors.md#_snippet_52

LANGUAGE: python
CODE:
```
from typing_extensions import Unpack

from pydantic import PydanticUserError, validate_call

try:

    @validate_call
    def func(**kwargs: Unpack[int]):
        pass

except PydanticUserError as exc_info:
    assert exc_info.code == 'unpack-typed-dict'
```

----------------------------------------

TITLE: Pydantic computed_field with Property for Volume Calculation
DESCRIPTION: Demonstrates using the computed_field decorator with a property to calculate the volume of a Box model. Includes JSON schema generation and model dumping.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/fields.md#_snippet_34

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, computed_field


class Box(BaseModel):
    width: float
    height: float
    depth: float

    @computed_field
    @property  # (1)!
    def volume(self) -> float:
        return self.width * self.height * self.depth


print(Box.model_json_schema(mode='serialization'))
```

LANGUAGE: json
CODE:
```
{
    'properties': {
        'width': {'title': 'Width', 'type': 'number'},
        'height': {'title': 'Height', 'type': 'number'},
        'depth': {'title': 'Depth', 'type': 'number'},
        'volume': {'readOnly': True, 'title': 'Volume', 'type': 'number'},
    },
    'required': ['width', 'height', 'depth', 'volume'],
    'title': 'Box',
    'type': 'object',
}
```

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, computed_field


class Box(BaseModel):
    width: float
    height: float
    depth: float

    @computed_field
    @property
    def volume(self) -> float:
        return self.width * self.height * self.depth


b = Box(width=1, height=2, depth=3)
print(b.model_dump())
```

----------------------------------------

TITLE: Mypy Plugin: Detect Required Fields
DESCRIPTION: Enhances the Pydantic mypy plugin's ability to accurately detect and report required fields within models. This leads to better static type checking and fewer runtime errors.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_79

LANGUAGE: python
CODE:
```
from pydantic import BaseModel

class User(BaseModel):
    id: int # Mypy plugin should correctly identify 'id' as required
    name: str = "Guest" # Mypy plugin should identify 'name' as optional

# Mypy check would flag missing 'id' if not provided:
# user = User(name="Alice") # This should raise a mypy error

```

----------------------------------------

TITLE: Performance: Optimize get_type_ref Calls
DESCRIPTION: Improves the performance of calls to `get_type_ref` by optimizing its internal implementation, contributing to faster schema building.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_31

LANGUAGE: python
CODE:
```
# Internal optimization for type referencing during schema generation.
```

----------------------------------------

TITLE: Support for re.Pattern Type
DESCRIPTION: Adds native support for the `re.Pattern` type, allowing Pydantic models to directly validate and handle compiled regular expression objects.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_73

LANGUAGE: python
CODE:
```
import re
from pydantic import BaseModel

class RegexModel(BaseModel):
    pattern: re.Pattern

compiled_regex = re.compile(r'^\d+$')
model = RegexModel(pattern=compiled_regex)

print(model.pattern)

```

----------------------------------------

TITLE: Pydantic v0.32.2: Dataclass Inheritance and GenericModels
DESCRIPTION: Fixes related to `__post_init__` usage with dataclass inheritance and validation of required fields on GenericModels classes. Also addresses custom `Schema` definition on `GenericModel` fields.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_261

LANGUAGE: APIDOC
CODE:
```
Pydantic v0.32.2 Release Notes:

- Fix `__post_init__` usage with dataclass inheritance.
  - Related to issue [#739](https://github.com/pydantic/pydantic/pull/739).
- Fix required fields validation on GenericModels classes.
  - Related to issue [#742](https://github.com/pydantic/pydantic/pull/742).
- Fix defining custom `Schema` on `GenericModel` fields.
  - Related to issue [#754](https://github.com/pydantic/pydantic/pull/754).
```

----------------------------------------

TITLE: Pydantic Serialization by Alias (Runtime Flag)
DESCRIPTION: Enables serialization of model fields using their defined aliases on a per-call basis using the `by_alias=True` flag in `model_dump()` or `model_dump_json()`.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/alias.md#_snippet_15

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field


class Model(BaseModel):
    my_field: str = Field(serialization_alias='my_alias')


m = Model(my_field='foo')
print(m.model_dump(by_alias=True))
#> {'my_alias': 'foo'}
```

----------------------------------------

TITLE: Define Pydantic Model with Nested Types
DESCRIPTION: Demonstrates defining a Pydantic BaseModel that inherits from another class (Base) and uses various type annotations, including local types and forward references. Shows how Pydantic handles complex type resolution.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/internals/resolving_annotations.md#_snippet_3

LANGUAGE: python
CODE:
```
from pydantic import BaseModel

from module1 import Base

type MyType = str


def inner() -> None:
    type InnerType = bool

    class Model(BaseModel, Base):
        type LocalType = bytes

        f2: 'MyType'
        f3: 'InnerType'
        f4: 'LocalType'
        f5: 'UnknownType'

    type InnerType2 = complex
```

----------------------------------------

TITLE: Address Mypy Plugin Bugs
DESCRIPTION: Resolves bugs within the Pydantic mypy plugin, specifically addressing issues caused by `explicit_package_bases=True` and ensuring implicit defaults are added for Fields without default arguments.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_57

LANGUAGE: python
CODE:
```
# Address bug in mypy plugin caused by explicit_package_bases=True, [#5191](https://github.com/pydantic/pydantic/pull/5191) by @dmontagu
# Add implicit defaults in the mypy plugin for Field with no default argument, [#5190](https://github.com/pydantic/pydantic/pull/5190) by @dmontagu
```

----------------------------------------

TITLE: Pydantic: Ignore Extra Data
DESCRIPTION: Demonstrates Pydantic's default behavior of ignoring extra fields provided during model instantiation. Extra data is simply discarded.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md#_snippet_10

LANGUAGE: python
CODE:
```
from pydantic import BaseModel


class Model(BaseModel):
    x: int


m = Model(x=1, y='a')
assert m.model_dump() == {'x': 1}
```

----------------------------------------

TITLE: Avoid Primitive Subclasses
DESCRIPTION: Subclassing primitive types like `str` to add custom behavior can complicate Pydantic's validation and serialization. It's often cleaner and more performant to use Pydantic models to represent structured data with custom attributes.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/performance.md#_snippet_2

LANGUAGE: python
CODE:
```
# Don't do this: Subclassing str for custom attributes
class CompletedStr(str):
    def __init__(self, s: str):
        self.s = s
        self.done = False


# Do this: Use a Pydantic model for structured data
from pydantic import BaseModel


class CompletedModel(BaseModel):
    s: str
    done: bool = False


# Example usage:
# completed_model = CompletedModel(s='task', done=True)
```

----------------------------------------

TITLE: Pydantic Field Customization API
DESCRIPTION: Provides an overview of the pydantic.fields.Field function for customizing Pydantic model fields, including default values, JSON Schema metadata, and constraints.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/concepts/fields.md#_snippet_5

LANGUAGE: APIDOC
CODE:
```
pydantic.fields.Field

Used to customize Pydantic model fields, providing mechanisms for default values, JSON Schema metadata, constraints, and more. It behaves similarly to dataclasses.field.

Usage:
```python
from pydantic import BaseModel, Field

class Model(BaseModel):
    name: str = Field(frozen=True)
    # Or to explicitly mark as required:
    # name: str = Field(..., frozen=True)
```

Key Parameters and Concepts:
- `default`: Sets a default value for the field. If not provided and the field is not marked with `...` (ellipsis), it's considered optional.
- `default_factory`: A callable that returns a default value. Useful for mutable defaults like lists or dictionaries.
- `alias`: Specifies a different name for the field when parsing JSON.
- `frozen`: If True, the field cannot be changed after initialization.
- `...` (Ellipsis): Used to explicitly mark a field as required, even if a default value is assigned.
- `Annotated`: Can be used with `Field` to attach metadata to a field's type, allowing for more complex validation and customization.

Example with Annotated:
```python
from typing import Annotated
from pydantic import BaseModel, Field, WithJsonSchema

class Model(BaseModel):
    name: Annotated[str, Field(strict=True), WithJsonSchema({'extra': 'data'})]
```

Considerations:
- Arguments like `default`, `default_factory`, and `alias` are recognized by static type checkers for `__init__` synthesis.
- The `Annotated` pattern is not directly understood by static type checkers for `__init__` synthesis, so use the normal assignment form when type checker compatibility is critical for these arguments.
- Metadata applied via `Annotated` can be specific to the type or the field. Ensure correct placement to achieve the desired effect (e.g., `deprecated` flag applies to the field when applied to the top-level type).
```

----------------------------------------

TITLE: Pydantic v2.11.0 Fixes
DESCRIPTION: Details fixes in Pydantic v2.11.0, including handling generic typed dictionaries for unpacked variadic keyword parameters, resolving runtime errors with model string representations involving cached properties, preserving pipeline steps with ellipsis, and fixing deferred discriminator application logic.

SOURCE: https://github.com/pydantic/pydantic/blob/main/HISTORY.md#_snippet_10

LANGUAGE: python
CODE:
```
# Fixes in v2.11.0
# Allow generic typed dictionaries to be used for unpacked variadic keyword parameters
# Fix runtime error when computing model string representation involving cached properties and self-referenced models
# Preserve other steps when using the ellipsis in the pipeline API
# Fix deferred discriminator application logic
```

----------------------------------------

TITLE: Pydantic Custom Root Types with RootModel
DESCRIPTION: Introduces `RootModel` in Pydantic V2 as the successor to the `__root__` field for defining custom root types. This provides a more explicit and robust way to handle models that wrap a single value.

SOURCE: https://github.com/pydantic/pydantic/blob/main/docs/migration.md#_snippet_12

LANGUAGE: APIDOC
CODE:
```
Pydantic V2 Custom Root Types:

- `RootModel` replaces the `__root__` field from Pydantic V1.
- Purpose: To define models that wrap a single value or a collection.
- Usage: Inherit from `RootModel` and specify the root type as a generic parameter.
- Configuration Note: `RootModel` types no longer support the `arbitrary_types_allowed` config setting.

Example Usage:
```python
from pydantic import BaseModel, RootModel
from typing import List

# V1 style (deprecated)
# class MyRootModelV1(BaseModel):
#     __root__: List[int]

# V2 style using RootModel
class MyRootModelV2(RootModel[List[int]]):
    pass

# Validation and usage
root_instance = MyRootModelV2([1, 2, 3])
print(root_instance.root_value) # Access the wrapped value
print(root_instance.model_dump())
# Expected output: {'root': [1, 2, 3]}

# Validation with different types
# invalid_instance = MyRootModelV2([1, 'a', 3]) # This would raise a ValidationError
```
```