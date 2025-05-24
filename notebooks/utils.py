import uuid

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import render_text_description
from langchain_ollama import ChatOllama


class JsonOrRawParser(JsonOutputParser):
    def invoke(self, input, config=None):
        try:
            tool_calls = super().invoke(input)
            tool_calls = [
                tool_call | {"id": str(uuid.uuid4()), "type": "tool_call"}
                for tool_call in tool_calls
            ]
            input.tool_calls = tool_calls
            input.content = ""

            return input

        except Exception as e:
            return input


class ChatOllamaCustomized(ChatOllama):
    def bind_tools(self, tools):

        system_template = """\
        You are a helpful assistant with access to the following powerful tools:

        [{rendered_tools}]

        **Crucial Instruction:**
        If a user's request can be answered by using any of the available tools, you MUST use the tool(s) to retrieve the information.
        Only if no tool is suitable, you may respond directly.

        When using a tool, you MUST respond ONLY with the tool invocation(s) in the exact JSON format below. Do NOT include any other text, explanation, or commentary outside this JSON structure:

        [
          {{{{
            "name": "tool_name",
            "arguments": {{{{
              "arg1": "value1",
              "arg2": "value2"
            }}}}
          }}}},
          ...
        ]

        If no tool is appropriate for the user's query, you may respond normally with a direct answer.
        """

        rendered_tools = []
        for tool in tools:
            rendered_tools.append(
                json.dumps(tool.args_schema.model_json_schema(), indent=2)
            )
        rendered_tools_str = ",\n".join(rendered_tools)
        rendered_tools_str = rendered_tools_str.replace("{", "{{").replace("}", "}}")

        system_prompt_template = SystemMessagePromptTemplate.from_template(
            system_template
        )
        system_prompt = system_prompt_template.format_messages(
            rendered_tools=rendered_tools_str
        )

        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt[0].content), ("user", "{input}")]
        )

        chain = prompt | self | JsonOrRawParser()

        return chain.bind()
