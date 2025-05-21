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
        rendered_tools = render_text_description(tools)
        system_prompt = f"""\
        You have access to functions. If you decide to invoke any of the function(s),
        you MUST put it in the format of
        json[
          {{{{
            "name": "tool_name",
            "arguments": dictionary of argument name and its value
          }}}},
          ...
        ]
        You SHOULD NOT include any other text in the response if you call a function.
        If you have access to retrieval functions use it preferably.
        {rendered_tools}
        """
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("user", "{input}")]
        )
        chain = prompt | self | JsonOrRawParser()

        return chain.bind()
