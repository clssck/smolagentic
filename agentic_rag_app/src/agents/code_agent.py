"""
Code Agent

Specialized agent for code generation, programming tasks, and technical implementation.
"""

import re
import ast
from typing import Dict, Any, List
from smolagents import CodeAgent as SmolagentsCodeAgent, LiteLLMModel
from .base_agent import BaseAgent


class CodeAgent(BaseAgent):
    """Agent specialized for code generation and programming tasks"""
    
    def __init__(self, **kwargs):
        """Initialize code agent"""
        
        # Set defaults
        config = {
            "name": "code_agent",
            "model_id": "openrouter/mistralai/mistral-small-3.2-24b-instruct",
            "tools": [],  # CodeAgent has built-in tools
            "max_steps": 10,
            "temperature": 0.1,
            "max_tokens": 2000,
            "description": "Code generation, programming, and technical implementation"
        }
        
        # Update with provided kwargs
        config.update(kwargs)
        
        super().__init__(**config)
    
    def can_handle(self, query: str, context: Dict[str, Any] = None) -> bool:
        """
        Determine if this agent can handle the query
        
        Code agent handles:
        - Code generation requests
        - Programming questions
        - Algorithm implementations
        - Code reviews and debugging
        - Technical implementations
        """
        code_keywords = [
            "code", "program", "function", "class", "implement", "algorithm",
            "python", "javascript", "java", "c++", "sql", "html", "css",
            "debug", "fix", "optimize", "refactor", "test", "unit test",
            "api", "endpoint", "database", "schema", "query", "script",
            "write", "create", "build", "develop", "generate"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in code_keywords)
    
    def get_system_prompt(self) -> str:
        """Get code agent system prompt"""
        return """You are a Code Agent specialized in programming and software development.

Your capabilities:
- Generate high-quality code in multiple programming languages
- Implement algorithms and data structures
- Debug and fix code issues
- Optimize code performance
- Create unit tests and documentation
- Review code for best practices
- Design APIs and database schemas
- Write scripts and automation tools

Instructions:
1. Always provide clean, readable, and well-commented code
2. Follow language-specific best practices and conventions
3. Include error handling where appropriate
4. Provide explanations for complex logic
5. Suggest improvements and optimizations
6. Include usage examples when helpful
7. Consider security implications

Focus on:
- Code quality and maintainability
- Performance and efficiency
- Security best practices
- Proper error handling
- Clear documentation
- Testing considerations"""
    
    def _create_agent(self) -> SmolagentsCodeAgent:
        """Create the underlying code agent instance"""
        # Return the smolagents CodeAgent directly
        return SmolagentsCodeAgent(
            tools=[],
            model=self.model,
            max_steps=self.max_steps
        )
    
    def run(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute code generation/programming query
        """
        import time
        start_time = time.time()
        context = context or {}
        
        try:
            self.logger.info(f"Processing code query: {query[:100]}...")
            
            # Enhance query with code-specific context
            enhanced_query = self._enhance_code_query(query)
            
            # Use the specialized code agent
            response = self.agent.run(enhanced_query)
            
            # Analyze the response
            code_analysis = self._analyze_code_response(str(response))
            
            execution_time = time.time() - start_time
            
            result = {
                "response": response,
                "agent_name": self.name,
                "execution_time": execution_time,
                "model_used": self.model_id,
                "tools_used": ["code_execution", "python_interpreter"],
                "success": True,
                "error": None,
                "code_analysis": code_analysis
            }
            
            self.logger.info(f"Code query processed successfully in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            self.logger.error(f"Error processing code query: {error_msg}")
            
            return {
                "response": f"Error: {error_msg}",
                "agent_name": self.name,
                "execution_time": execution_time,
                "model_used": self.model_id,
                "tools_used": [],
                "success": False,
                "error": error_msg,
                "code_analysis": {}
            }
    
    def _enhance_code_query(self, query: str) -> str:
        """Enhance query with code-specific context"""
        # Detect programming language
        language = self._detect_language(query)
        
        # Add context based on query type
        if any(word in query.lower() for word in ["implement", "create", "write", "build"]):
            enhancement = f"""Code Implementation Request: {query}

Please provide:
1. Clean, well-commented code
2. Proper error handling
3. Usage examples
4. Brief explanation of the approach
"""
        elif any(word in query.lower() for word in ["debug", "fix", "error", "issue"]):
            enhancement = f"""Code Debugging Request: {query}

Please:
1. Identify the issue
2. Provide the corrected code
3. Explain what was wrong
4. Suggest prevention strategies
"""
        elif any(word in query.lower() for word in ["optimize", "improve", "performance"]):
            enhancement = f"""Code Optimization Request: {query}

Please:
1. Analyze the current code
2. Identify optimization opportunities
3. Provide optimized version
4. Explain the improvements
"""
        else:
            enhancement = f"""Programming Query: {query}

Please provide a comprehensive response with code examples where applicable."""
        
        if language:
            enhancement += f"\n\nPreferred Language: {language}"
        
        return enhancement
    
    def _detect_language(self, query: str) -> str:
        """Detect programming language from query"""
        languages = {
            "python": ["python", "py", "pandas", "numpy", "django", "flask"],
            "javascript": ["javascript", "js", "node", "react", "vue", "angular"],
            "java": ["java", "spring", "hibernate"],
            "c++": ["c++", "cpp", "c plus plus"],
            "sql": ["sql", "database", "query", "select", "insert", "update"],
            "html": ["html", "web", "webpage"],
            "css": ["css", "style", "styling"],
            "bash": ["bash", "shell", "script", "command line"]
        }
        
        query_lower = query.lower()
        for lang, keywords in languages.items():
            if any(keyword in query_lower for keyword in keywords):
                return lang
        
        return None
    
    def _analyze_code_response(self, response: str) -> Dict[str, Any]:
        """Analyze the code response for metadata"""
        analysis = {
            "contains_code": False,
            "languages_detected": [],
            "code_blocks": 0,
            "functions_defined": 0,
            "classes_defined": 0,
            "imports_used": []
        }
        
        # Check for code blocks
        code_block_pattern = r'```(\w+)?\n(.*?)\n```'
        code_blocks = re.findall(code_block_pattern, response, re.DOTALL)
        
        if code_blocks:
            analysis["contains_code"] = True
            analysis["code_blocks"] = len(code_blocks)
            
            for lang, code in code_blocks:
                if lang:
                    analysis["languages_detected"].append(lang)
                
                # Try to parse Python code for additional analysis
                if lang == "python" or not lang:
                    try:
                        tree = ast.parse(code)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef):
                                analysis["functions_defined"] += 1
                            elif isinstance(node, ast.ClassDef):
                                analysis["classes_defined"] += 1
                            elif isinstance(node, ast.Import):
                                for alias in node.names:
                                    analysis["imports_used"].append(alias.name)
                            elif isinstance(node, ast.ImportFrom):
                                if node.module:
                                    analysis["imports_used"].append(node.module)
                    except:
                        pass  # Ignore parsing errors
        
        # Remove duplicates
        analysis["languages_detected"] = list(set(analysis["languages_detected"]))
        analysis["imports_used"] = list(set(analysis["imports_used"]))
        
        return analysis
    
    def execute_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """
        Execute code safely (if supported)
        
        Args:
            code: Code to execute
            language: Programming language
            
        Returns:
            Execution results
        """
        try:
            if language.lower() == "python":
                # Use the code agent's execution capabilities
                execution_prompt = f"""Execute this Python code and return the results:

```python
{code}
```

Please run the code and show the output."""
                
                result = self.agent.run(execution_prompt)
                
                return {
                    "success": True,
                    "output": str(result),
                    "language": language,
                    "code": code
                }
            else:
                return {
                    "success": False,
                    "error": f"Code execution not supported for language: {language}",
                    "language": language,
                    "code": code
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "language": language,
                "code": code
            }
    
    def review_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """
        Review code for best practices and improvements
        
        Args:
            code: Code to review
            language: Programming language
            
        Returns:
            Code review results
        """
        review_prompt = f"""Please review this {language} code for:
1. Code quality and best practices
2. Security issues
3. Performance optimizations
4. Bug potential
5. Maintainability improvements

Code to review:
```{language}
{code}
```

Provide a detailed code review with specific suggestions."""
        
        try:
            review_result = self.agent.run(review_prompt)
            
            return {
                "success": True,
                "review": str(review_result),
                "language": language,
                "code": code
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "language": language,
                "code": code
            }