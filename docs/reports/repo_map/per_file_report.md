# Per-File Analysis Report

## File: root/hlynr_intercept/__init__.py

### Imports (resolved)
- (none)

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/hlynr_intercept/envs/__init__.py

### Imports (resolved)
- (none)

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/hlynr_intercept/envs/debug_render.py

### Imports (resolved)
- (none)

### Exports (symbols)
- functions: ['minimal_render_test']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: False

### Notes

---

## File: root/hlynr_intercept/envs/demo.py

### Imports (resolved)
- (none)

### Exports (symbols)
- functions: ['run_manual_demo', 'run_api_demo', 'run_rl_example']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: False

### Notes

---

## File: root/hlynr_intercept/envs/demo_simple.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['Missile', 'InterceptWindow']
- functions: ['unit', 'euler_to_quat', 'quat_to_mat4']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: False

### Notes

---

## File: root/hlynr_intercept/envs/sim3d/__init__.py

### Imports (resolved)
- root/hlynr_intercept/envs/sim3d/interface/api.py [import style: relative, symbol(s): MissileInterceptSim]

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/hlynr_intercept/envs/sim3d/core/__init__.py

### Imports (resolved)
- root/hlynr_intercept/envs/sim3d/core/physics.py [import style: relative, symbol(s): Physics6DOF]
- root/hlynr_intercept/envs/sim3d/core/missiles.py [import style: relative, symbol(s): Missile]
- root/hlynr_intercept/envs/sim3d/core/world.py [import style: relative, symbol(s): World]

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/hlynr_intercept/envs/sim3d/core/missiles.py

### Imports (resolved)
- root/hlynr_intercept/envs/sim3d/core/physics.py [import style: relative, symbol(s): Physics6DOF]

### Exports (symbols)
- classes: ['Missile']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/hlynr_intercept/envs/sim3d/core/physics.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['Physics6DOF']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/hlynr_intercept/envs/sim3d/core/world.py

### Imports (resolved)
- root/hlynr_intercept/envs/sim3d/core/physics.py [import style: relative, symbol(s): Physics6DOF]
- root/hlynr_intercept/envs/sim3d/core/missiles.py [import style: relative, symbol(s): Missile]

### Exports (symbols)
- classes: ['World']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/hlynr_intercept/envs/sim3d/interface/__init__.py

### Imports (resolved)
- root/hlynr_intercept/envs/sim3d/interface/api.py [import style: relative, symbol(s): MissileInterceptSim]
- root/hlynr_intercept/envs/sim3d/interface/controls.py [import style: relative, symbol(s): ManualControls]

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/hlynr_intercept/envs/sim3d/interface/api.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['MissileInterceptSim']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/hlynr_intercept/envs/sim3d/interface/controls.py

### Imports (resolved)
- root/hlynr_intercept/envs/sim3d/interface/api.py [import style: relative, symbol(s): MissileInterceptSim]

### Exports (symbols)
- classes: ['ManualControls']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/hlynr_intercept/envs/sim3d/render/__init__.py

### Imports (resolved)
- root/hlynr_intercept/envs/sim3d/render/engine.py [import style: relative, symbol(s): RenderEngine]
- root/hlynr_intercept/envs/sim3d/render/camera.py [import style: relative, symbol(s): Camera]

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/hlynr_intercept/envs/sim3d/render/camera.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['Camera']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/hlynr_intercept/envs/sim3d/render/engine.py

### Imports (resolved)
- root/hlynr_intercept/envs/sim3d/render/camera.py [import style: relative, symbol(s): Camera]

### Exports (symbols)
- classes: ['RenderEngine']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/hlynr_intercept/envs/sim3d/sensors/__init__.py

### Imports (resolved)
- root/hlynr_intercept/envs/sim3d/sensors/radar.py [import style: relative, symbol(s): GroundRadar, InterceptorRadar, RadarContact]
- root/hlynr_intercept/envs/sim3d/sensors/scope.py [import style: relative, symbol(s): RadarScope]

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/hlynr_intercept/envs/sim3d/sensors/radar.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['RadarContact', 'GroundRadar', 'InterceptorRadar']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/hlynr_intercept/envs/sim3d/sensors/scope.py

### Imports (resolved)
- root/hlynr_intercept/envs/sim3d/sensors/radar.py [import style: relative, symbol(s): RadarContact, GroundRadar, InterceptorRadar]

### Exports (symbols)
- classes: ['RadarScope']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/hlynr_intercept/envs/test_fixes.py

### Imports (resolved)
- (none)

### Exports (symbols)
- functions: ['test_headless_simulation', 'test_rendering_components', 'test_coordinate_scaling']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: False

### Notes

---

## File: root/hlynr_intercept/training/__init__.py

### Imports (resolved)
- (none)

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/hlynr_intercept/utils/__init__.py

### Imports (resolved)
- (none)

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/scripts/agno_agent.py

### Imports (resolved)
- (none)

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: False

### Notes

---

## File: root/serena/scripts/demo_run_tools.py

### Imports (resolved)
- (none)

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: False

### Notes

---

## File: root/serena/scripts/gen_prompt_factory.py

### Imports (resolved)
- (none)

### Exports (symbols)
- functions: ['main']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: False

### Notes

---

## File: root/serena/scripts/mcp_server.py

### Imports (resolved)
- (none)

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: False

### Notes
- HTTP/API server implementation

---

## File: root/serena/scripts/print_mode_context_options.py

### Imports (resolved)
- (none)

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: False

### Notes

---

## File: root/serena/scripts/print_tool_overview.py

### Imports (resolved)
- (none)

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: False

### Notes

---

## File: root/serena/src/interprompt/__init__.py

### Imports (resolved)
- root/serena/src/interprompt/prompt_factory.py [import style: relative, symbol(s): autogenerate_prompt_factory_module]

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/interprompt/jinja_template.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['ParameterizedTemplateInterface', '_JinjaEnvProvider', 'JinjaTemplate']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/interprompt/multilang_prompt.py

### Imports (resolved)
- root/serena/src/interprompt/jinja_template.py [import style: relative, symbol(s): JinjaTemplate, ParameterizedTemplateInterface]

### Exports (symbols)
- classes: ['PromptTemplate', 'PromptList', 'LanguageFallbackMode', '_MultiLangContainer', 'MultiLangPromptTemplate', 'MultiLangPromptList', 'MultiLangPromptCollection']
- variables: ['T', 'DEFAULT_LANG_CODE']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/interprompt/prompt_factory.py

### Imports (resolved)
- root/serena/src/interprompt/multilang_prompt.py [import style: relative, symbol(s): DEFAULT_LANG_CODE, LanguageFallbackMode, MultiLangPromptCollection, PromptList]

### Exports (symbols)
- classes: ['PromptFactoryBase']
- functions: ['autogenerate_prompt_factory_module']

### Exports (files/artifacts written or produced)
- <dynamic path>; callsite: <function:80>; method: open

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/interprompt/util/__init__.py

### Imports (resolved)
- (none)

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/interprompt/util/class_decorators.py

### Imports (resolved)
- (none)

### Exports (symbols)
- functions: ['singleton']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/__init__.py

### Imports (resolved)
- (none)

### Exports (symbols)
- functions: ['serena_version']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/agent.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['ProjectNotFoundError', 'LinesRead', 'MemoriesManager', 'AvailableTools', 'SerenaAgent']
- variables: ['T', 'SUCCESS_RESULT']

### Exports (files/artifacts written or produced)
- <dynamic path>; callsite: <function:83>; method: open

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/agno.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['SerenaAgnoToolkit', 'SerenaAgnoAgentProvider']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: True

### Notes

---

## File: root/serena/src/serena/analytics.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TokenCountEstimator', 'TiktokenCountEstimator', 'AnthropicTokenCount', 'RegisteredTokenCountEstimator', 'ToolUsageStats']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/cli.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['ProjectType', 'AutoRegisteringGroup', 'TopLevelCommands', 'ModeCommands', 'ContextCommands', 'SerenaConfigCommands', 'ProjectCommands', 'ToolCommands', 'PromptCommands']
- functions: ['_open_in_editor', 'get_help']
- variables: ['PROJECT_TYPE']

### Exports (files/artifacts written or produced)
- <dynamic path>; callsite: <function:477>; method: open

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/code_editor.py

### Imports (resolved)
- root/serena/src/serena/project.py [import style: relative, symbol(s): Project]
- root/serena/src/serena/tools/jetbrains_plugin_client.py [import style: relative, symbol(s): JetBrainsPluginClient]
- root/serena/src/serena/agent.py [import style: relative, symbol(s): SerenaAgent]

### Exports (symbols)
- classes: ['CodeEditor', 'LanguageServerCodeEditor', 'JetBrainsCodeEditor']

### Exports (files/artifacts written or produced)
- <dynamic path>; callsite: <function:61>; method: open

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/config/__init__.py

### Imports (resolved)
- (none)

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/config/context_mode.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['SerenaAgentMode', 'SerenaAgentContext', 'RegisteredContext', 'RegisteredMode']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/config/serena_config.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['SerenaPaths', 'ToolSet', 'ToolInclusionDefinition', 'SerenaConfigError', 'ProjectConfig', 'RegisteredProject', 'SerenaConfig']
- functions: ['get_serena_managed_in_project_dir', 'is_running_in_docker']
- variables: ['T']

### Exports (files/artifacts written or produced)
- <self>.save(...); callsite: <function:550>; method: save
- <self>.save(...); callsite: <function:562>; method: save
- <instance>.save(...); callsite: <function:468>; method: save
- <dynamic path>; callsite: <function:487>; method: open

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/constants.py

### Imports (resolved)
- (none)

### Exports (symbols)
- variables: ['SERENA_MANAGED_DIR_NAME', 'SERENA_MANAGED_DIR_IN_HOME', 'REPO_ROOT', 'PROMPT_TEMPLATES_DIR_INTERNAL', 'PROMPT_TEMPLATES_DIR_IN_USER_HOME', 'SERENAS_OWN_CONTEXT_YAMLS_DIR', 'USER_CONTEXT_YAMLS_DIR', 'SERENAS_OWN_MODE_YAMLS_DIR', 'USER_MODE_YAMLS_DIR', 'INTERNAL_MODE_YAMLS_DIR', 'SERENA_DASHBOARD_DIR', 'SERENA_ICON_DIR', 'DEFAULT_ENCODING', 'DEFAULT_CONTEXT', 'DEFAULT_MODES', 'PROJECT_TEMPLATE_FILE', 'SERENA_CONFIG_TEMPLATE_FILE', 'SERENA_LOG_FORMAT']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/dashboard.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['RequestLog', 'ResponseLog', 'ResponseToolNames', 'ResponseToolStats', 'SerenaDashboardAPI']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/generated/generated_prompt_factory.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['PromptFactory']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/gui_log_viewer.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['LogLevel', 'GuiLogViewer', 'GuiLogViewerHandler']
- functions: ['show_fatal_exception']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/mcp.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['SerenaMCPRequestContext', 'SerenaMCPFactory', 'SerenaMCPFactorySingleProcess']
- functions: ['configure_logging']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/project.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['Project']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/prompt_factory.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['SerenaPromptFactory']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/symbol.py

### Imports (resolved)
- root/serena/src/serena/project.py [import style: relative, symbol(s): Project]
- root/serena/src/serena/agent.py [import style: relative, symbol(s): SerenaAgent]

### Exports (symbols)
- classes: ['LanguageServerSymbolLocation', 'PositionInFile', 'Symbol', 'LanguageServerSymbol', 'ReferenceInLanguageServerSymbol', 'LanguageServerSymbolRetriever', 'JetBrainsSymbol']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/text_utils.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['LineType', 'TextLine', 'MatchedConsecutiveLines']
- functions: ['glob_to_regex', 'search_text', 'default_file_reader', 'glob_match', 'search_files']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/tools/__init__.py

### Imports (resolved)
- root/serena/src/serena/tools/tools_base.py [import style: relative, symbol(s): *]
- root/serena/src/serena/tools/file_tools.py [import style: relative, symbol(s): *]
- root/serena/src/serena/tools/symbol_tools.py [import style: relative, symbol(s): *]
- root/serena/src/serena/tools/memory_tools.py [import style: relative, symbol(s): *]
- root/serena/src/serena/tools/cmd_tools.py [import style: relative, symbol(s): *]
- root/serena/src/serena/tools/config_tools.py [import style: relative, symbol(s): *]
- root/serena/src/serena/tools/workflow_tools.py [import style: relative, symbol(s): *]
- root/serena/src/serena/tools/jetbrains_tools.py [import style: relative, symbol(s): *]

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/tools/cmd_tools.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['ExecuteShellCommandTool']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/tools/config_tools.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['ActivateProjectTool', 'RemoveProjectTool', 'SwitchModesTool', 'GetCurrentConfigTool']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/tools/file_tools.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['ReadFileTool', 'CreateTextFileTool', 'ListDirTool', 'FindFileTool', 'ReplaceRegexTool', 'DeleteLinesTool', 'ReplaceLinesTool', 'InsertAtLineTool', 'SearchForPatternTool']

### Exports (files/artifacts written or produced)
- <abs_path>.write_text(...); callsite: <function:78>; method: write_text

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/tools/jetbrains_plugin_client.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['SerenaClientError', 'ConnectionError', 'APIError', 'ServerNotFoundError', 'JetBrainsPluginClient']
- variables: ['T']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/tools/jetbrains_tools.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['JetBrainsFindSymbolTool', 'JetBrainsFindReferencingSymbolsTool', 'JetBrainsGetSymbolsOverviewTool']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/tools/memory_tools.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['WriteMemoryTool', 'ReadMemoryTool', 'ListMemoriesTool', 'DeleteMemoryTool']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/tools/symbol_tools.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['RestartLanguageServerTool', 'GetSymbolsOverviewTool', 'FindSymbolTool', 'FindReferencingSymbolsTool', 'ReplaceSymbolBodyTool', 'InsertAfterSymbolTool', 'InsertBeforeSymbolTool']
- functions: ['_sanitize_symbol_dict']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/tools/tools_base.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['Component', 'ToolMarker', 'ToolMarkerCanEdit', 'ToolMarkerDoesNotRequireActiveProject', 'ToolMarkerOptional', 'ToolMarkerSymbolicRead', 'ToolMarkerSymbolicEdit', 'ApplyMethodProtocol', 'Tool', 'EditedFileContext', 'RegisteredTool', 'ToolRegistry']
- variables: ['T', 'SUCCESS_RESULT']

### Exports (files/artifacts written or produced)
- <dynamic path>; callsite: <function:333>; method: open

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/tools/workflow_tools.py

### Imports (resolved)
- root/serena/src/serena/tools/memory_tools.py [import style: relative, symbol(s): ListMemoriesTool]

### Exports (symbols)
- classes: ['CheckOnboardingPerformedTool', 'OnboardingTool', 'ThinkAboutCollectedInformationTool', 'ThinkAboutTaskAdherenceTool', 'ThinkAboutWhetherYouAreDoneTool', 'SummarizeChangesTool', 'PrepareForNewConversationTool', 'InitialInstructionsTool']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/util/class_decorators.py

### Imports (resolved)
- (none)

### Exports (symbols)
- functions: ['singleton']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/util/exception.py

### Imports (resolved)
- (none)

### Exports (symbols)
- functions: ['is_headless_environment', 'show_fatal_exception_safe']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/util/file_system.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['ScanResult', 'GitignoreSpec', 'GitignoreParser']
- functions: ['scan_directory', 'find_all_non_ignored_files', 'match_path']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/util/general.py

### Imports (resolved)
- (none)

### Exports (symbols)
- functions: ['_create_YAML', 'load_yaml', 'load_yaml', 'load_yaml', 'save_yaml']

### Exports (files/artifacts written or produced)
- <dynamic path>; callsite: <function:31>; method: open
- <yaml>.dump(...); callsite: <function:32>; method: dump

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/util/git.py

### Imports (resolved)
- root/serena/src/serena/util/shell.py [import style: relative, symbol(s): subprocess_check_output]

### Exports (symbols)
- functions: ['get_git_status']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/util/inspection.py

### Imports (resolved)
- (none)

### Exports (symbols)
- functions: ['iter_subclasses', 'determine_programming_language_composition']
- variables: ['T']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/util/logging.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['MemoryLogHandler', 'LogBuffer']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/util/shell.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['ShellCommandResult']
- functions: ['execute_shell_command', 'subprocess_check_output']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/serena/util/thread.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TimeoutException', 'ExecutionResult']
- functions: ['execute_with_timeout']
- variables: ['T']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/solidlsp/__init__.py

### Imports (resolved)
- root/serena/src/solidlsp/ls.py [import style: relative, symbol(s): SolidLanguageServer]

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/solidlsp/language_servers/bash_language_server.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['BashLanguageServer']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes
- HTTP/API server implementation

---

## File: root/serena/src/solidlsp/language_servers/clangd_language_server.py

### Imports (resolved)
- root/serena/src/solidlsp/language_servers/common.py [import style: relative, symbol(s): RuntimeDependency, RuntimeDependencyCollection]

### Exports (symbols)
- classes: ['ClangdLanguageServer']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes
- HTTP/API server implementation

---

## File: root/serena/src/solidlsp/language_servers/clojure_lsp.py

### Imports (resolved)
- root/serena/src/solidlsp/language_servers/common.py [import style: relative, symbol(s): RuntimeDependency, RuntimeDependencyCollection]

### Exports (symbols)
- classes: ['ClojureLSP']
- functions: ['run_command', 'verify_clojure_cli']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes
- HTTP/API server implementation

---

## File: root/serena/src/solidlsp/language_servers/common.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['RuntimeDependency', 'RuntimeDependencyCollection']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes
- HTTP/API server implementation

---

## File: root/serena/src/solidlsp/language_servers/csharp_language_server.py

### Imports (resolved)
- root/serena/src/solidlsp/language_servers/common.py [import style: relative, symbol(s): RuntimeDependency]

### Exports (symbols)
- classes: ['CSharpLanguageServer']
- functions: ['breadth_first_file_scan', 'find_solution_or_project_file']
- variables: ['RUNTIME_DEPENDENCIES']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes
- HTTP/API server implementation

---

## File: root/serena/src/solidlsp/language_servers/dart_language_server.py

### Imports (resolved)
- root/serena/src/solidlsp/language_servers/common.py [import style: relative, symbol(s): RuntimeDependency, RuntimeDependencyCollection]

### Exports (symbols)
- classes: ['DartLanguageServer']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes
- HTTP/API server implementation

---

## File: root/serena/src/solidlsp/language_servers/eclipse_jdtls.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['RuntimeDependencyPaths', 'EclipseJDTLS']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes
- HTTP/API server implementation

---

## File: root/serena/src/solidlsp/language_servers/elixir_tools/__init__.py

### Imports (resolved)
- (none)

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes
- HTTP/API server implementation

---

## File: root/serena/src/solidlsp/language_servers/elixir_tools/elixir_tools.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['ElixirTools']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes
- HTTP/API server implementation

---

## File: root/serena/src/solidlsp/language_servers/erlang_language_server.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['ErlangLanguageServer']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes
- HTTP/API server implementation

---

## File: root/serena/src/solidlsp/language_servers/gopls.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['Gopls']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes
- HTTP/API server implementation

---

## File: root/serena/src/solidlsp/language_servers/intelephense.py

### Imports (resolved)
- root/serena/src/solidlsp/language_servers/common.py [import style: relative, symbol(s): RuntimeDependency, RuntimeDependencyCollection]

### Exports (symbols)
- classes: ['Intelephense']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes
- HTTP/API server implementation

---

## File: root/serena/src/solidlsp/language_servers/jedi_server.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['JediServer']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes
- HTTP/API server implementation

---

## File: root/serena/src/solidlsp/language_servers/kotlin_language_server.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['KotlinRuntimeDependencyPaths', 'KotlinLanguageServer']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes
- HTTP/API server implementation

---

## File: root/serena/src/solidlsp/language_servers/lua_ls.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['LuaLanguageServer']

### Exports (files/artifacts written or produced)
- <dynamic path>; callsite: <function:114>; method: open

### Entry Points
- main guard: False
- argparse: False

### Notes
- HTTP/API server implementation

---

## File: root/serena/src/solidlsp/language_servers/nixd_ls.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['NixLanguageServer']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes
- HTTP/API server implementation

---

## File: root/serena/src/solidlsp/language_servers/omnisharp.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['OmniSharp']
- functions: ['breadth_first_file_scan', 'find_least_depth_sln_file']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes
- HTTP/API server implementation

---

## File: root/serena/src/solidlsp/language_servers/pyright_server.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['PyrightServer']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes
- HTTP/API server implementation

---

## File: root/serena/src/solidlsp/language_servers/rust_analyzer.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['RustAnalyzer']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes
- HTTP/API server implementation

---

## File: root/serena/src/solidlsp/language_servers/solargraph.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['Solargraph']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes
- HTTP/API server implementation

---

## File: root/serena/src/solidlsp/language_servers/sourcekit_lsp.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['SourceKitLSP']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes
- HTTP/API server implementation

---

## File: root/serena/src/solidlsp/language_servers/terraform_ls.py

### Imports (resolved)
- root/serena/src/solidlsp/language_servers/common.py [import style: relative, symbol(s): RuntimeDependency, RuntimeDependencyCollection]

### Exports (symbols)
- classes: ['TerraformLS']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes
- HTTP/API server implementation

---

## File: root/serena/src/solidlsp/language_servers/typescript_language_server.py

### Imports (resolved)
- root/serena/src/solidlsp/language_servers/common.py [import style: relative, symbol(s): RuntimeDependency, RuntimeDependencyCollection]

### Exports (symbols)
- classes: ['TypeScriptLanguageServer']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes
- HTTP/API server implementation

---

## File: root/serena/src/solidlsp/language_servers/vts_language_server.py

### Imports (resolved)
- root/serena/src/solidlsp/language_servers/common.py [import style: relative, symbol(s): RuntimeDependency, RuntimeDependencyCollection]

### Exports (symbols)
- classes: ['VtsLanguageServer']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes
- HTTP/API server implementation

---

## File: root/serena/src/solidlsp/language_servers/zls.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['ZigLanguageServer']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes
- HTTP/API server implementation

---

## File: root/serena/src/solidlsp/ls.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['ReferenceInSymbol', 'LSPFileBuffer', 'SolidLanguageServer']

### Exports (files/artifacts written or produced)
- <dynamic path>; callsite: <function:1643>; method: open
- <pickle>.dump(...); callsite: <function:1644>; method: dump

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/solidlsp/ls_config.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['FilenameMatcher', 'Language', 'LanguageServerConfig']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/solidlsp/ls_exceptions.py

### Imports (resolved)
- root/serena/src/solidlsp/ls_handler.py [import style: relative, symbol(s): LanguageServerTerminatedException]

### Exports (symbols)
- classes: ['SolidLSPException']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/solidlsp/ls_handler.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['LanguageServerTerminatedException', 'Request', 'SolidLanguageServerHandler']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/solidlsp/ls_logger.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['LogLine', 'LanguageServerLogger']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/solidlsp/ls_request.py

### Imports (resolved)
- root/serena/src/solidlsp/ls_handler.py [import style: relative, symbol(s): SolidLanguageServerHandler]

### Exports (symbols)
- classes: ['LanguageServerRequest']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/solidlsp/ls_types.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['Position', 'Range', 'Location', 'CompletionItemKind', 'CompletionItem', 'SymbolKind', 'SymbolTag', 'UnifiedSymbolInformation', 'MarkupKind', '__MarkedString_Type_1', 'MarkupContent', 'Hover', 'DiagnosticsSeverity', 'Diagnostic']
- variables: ['URI']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/solidlsp/ls_utils.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['InvalidTextLocationError', 'TextUtils', 'PathUtils', 'FileUtils', 'PlatformId', 'DotnetVersion', 'PlatformUtils', 'SymbolUtils']

### Exports (files/artifacts written or produced)
- <dynamic path>; callsite: <function:192>; method: open
- <dynamic path>; callsite: <function:216>; method: open
- <dynamic path>; callsite: <function:220>; method: open

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/solidlsp/lsp_protocol_handler/lsp_constants.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['LSPConstants']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/solidlsp/lsp_protocol_handler/lsp_requests.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['LspRequest', 'LspNotification']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/solidlsp/lsp_protocol_handler/lsp_types.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['SemanticTokenTypes', 'SemanticTokenModifiers', 'DocumentDiagnosticReportKind', 'ErrorCodes', 'LSPErrorCodes', 'FoldingRangeKind', 'SymbolKind', 'SymbolTag', 'UniquenessLevel', 'MonikerKind', 'InlayHintKind', 'MessageType', 'TextDocumentSyncKind', 'TextDocumentSaveReason', 'CompletionItemKind', 'CompletionItemTag', 'InsertTextFormat', 'InsertTextMode', 'DocumentHighlightKind', 'CodeActionKind', 'TraceValues', 'MarkupKind', 'PositionEncodingKind', 'FileChangeType', 'WatchKind', 'DiagnosticSeverity', 'DiagnosticTag', 'CompletionTriggerKind', 'SignatureHelpTriggerKind', 'CodeActionTriggerKind', 'FileOperationPatternKind', 'NotebookCellKind', 'ResourceOperationKind', 'FailureHandlingKind', 'PrepareSupportDefaultBehavior', 'TokenFormat', 'ImplementationParams', 'Location', 'ImplementationRegistrationOptions', 'TypeDefinitionParams', 'TypeDefinitionRegistrationOptions', 'WorkspaceFolder', 'DidChangeWorkspaceFoldersParams', 'ConfigurationParams', 'DocumentColorParams', 'ColorInformation', 'DocumentColorRegistrationOptions', 'ColorPresentationParams', 'ColorPresentation', 'WorkDoneProgressOptions', 'TextDocumentRegistrationOptions', 'FoldingRangeParams', 'FoldingRange', 'FoldingRangeRegistrationOptions', 'DeclarationParams', 'DeclarationRegistrationOptions', 'SelectionRangeParams', 'SelectionRange', 'SelectionRangeRegistrationOptions', 'WorkDoneProgressCreateParams', 'WorkDoneProgressCancelParams', 'CallHierarchyPrepareParams', 'CallHierarchyItem', 'CallHierarchyRegistrationOptions', 'CallHierarchyIncomingCallsParams', 'CallHierarchyOutgoingCallsParams', 'CallHierarchyOutgoingCall', 'SemanticTokensParams', 'SemanticTokens', 'SemanticTokensPartialResult', 'SemanticTokensRegistrationOptions', 'SemanticTokensDeltaParams', 'SemanticTokensDelta', 'SemanticTokensDeltaPartialResult', 'SemanticTokensRangeParams', 'ShowDocumentParams', 'ShowDocumentResult', 'LinkedEditingRangeParams', 'LinkedEditingRanges', 'LinkedEditingRangeRegistrationOptions', 'CreateFilesParams', 'WorkspaceEdit', 'FileOperationRegistrationOptions', 'RenameFilesParams', 'DeleteFilesParams', 'MonikerParams', 'Moniker', 'MonikerRegistrationOptions', 'TypeHierarchyPrepareParams', 'TypeHierarchyItem', 'TypeHierarchyRegistrationOptions', 'TypeHierarchySupertypesParams', 'TypeHierarchySubtypesParams', 'InlineValueParams', 'InlineValueRegistrationOptions', 'InlayHintParams', 'InlayHint', 'InlayHintRegistrationOptions', 'DocumentDiagnosticParams', 'DocumentDiagnosticReportPartialResult', 'DiagnosticServerCancellationData', 'DiagnosticRegistrationOptions', 'WorkspaceDiagnosticParams', 'WorkspaceDiagnosticReport', 'WorkspaceDiagnosticReportPartialResult', 'DidOpenNotebookDocumentParams', 'DidChangeNotebookDocumentParams', 'DidSaveNotebookDocumentParams', 'DidCloseNotebookDocumentParams', 'RegistrationParams', 'UnregistrationParams', 'InitializeParams', 'InitializeResult', 'InitializeError', 'InitializedParams', 'DidChangeConfigurationParams', 'DidChangeConfigurationRegistrationOptions', 'ShowMessageParams', 'ShowMessageRequestParams', 'MessageActionItem', 'LogMessageParams', 'DidOpenTextDocumentParams', 'DidChangeTextDocumentParams', 'TextDocumentChangeRegistrationOptions', 'DidCloseTextDocumentParams', 'DidSaveTextDocumentParams', 'TextDocumentSaveRegistrationOptions', 'WillSaveTextDocumentParams', 'TextEdit', 'DidChangeWatchedFilesParams', 'DidChangeWatchedFilesRegistrationOptions', 'PublishDiagnosticsParams', 'CompletionParams', 'CompletionItem', 'CompletionList', 'CompletionRegistrationOptions', 'HoverParams', 'Hover', 'HoverRegistrationOptions', 'SignatureHelpParams', 'SignatureHelp', 'SignatureHelpRegistrationOptions', 'DefinitionParams', 'DefinitionRegistrationOptions', 'ReferenceParams', 'ReferenceRegistrationOptions', 'DocumentHighlightParams', 'DocumentHighlight', 'DocumentHighlightRegistrationOptions', 'DocumentSymbolParams', 'SymbolInformation', 'DocumentSymbol', 'DocumentSymbolRegistrationOptions', 'CodeActionParams', 'Command', 'CodeAction', 'CodeActionRegistrationOptions', 'WorkspaceSymbolParams', 'WorkspaceSymbol', 'WorkspaceSymbolRegistrationOptions', 'CodeLensParams', 'CodeLens', 'CodeLensRegistrationOptions', 'DocumentLinkParams', 'DocumentLink', 'DocumentLinkRegistrationOptions', 'DocumentFormattingParams', 'DocumentFormattingRegistrationOptions', 'DocumentRangeFormattingParams', 'DocumentRangeFormattingRegistrationOptions', 'DocumentOnTypeFormattingParams', 'DocumentOnTypeFormattingRegistrationOptions', 'RenameParams', 'RenameRegistrationOptions', 'PrepareRenameParams', 'ExecuteCommandParams', 'ExecuteCommandRegistrationOptions', 'ApplyWorkspaceEditParams', 'ApplyWorkspaceEditResult', 'WorkDoneProgressBegin', 'WorkDoneProgressReport', 'WorkDoneProgressEnd', 'SetTraceParams', 'LogTraceParams', 'CancelParams', 'ProgressParams', 'TextDocumentPositionParams', 'WorkDoneProgressParams', 'PartialResultParams', 'LocationLink', 'Range', 'ImplementationOptions', 'StaticRegistrationOptions', 'TypeDefinitionOptions', 'WorkspaceFoldersChangeEvent', 'ConfigurationItem', 'TextDocumentIdentifier', 'Color', 'DocumentColorOptions', 'FoldingRangeOptions', 'DeclarationOptions', 'Position', 'SelectionRangeOptions', 'CallHierarchyOptions', 'SemanticTokensOptions', 'SemanticTokensEdit', 'LinkedEditingRangeOptions', 'FileCreate', 'TextDocumentEdit', 'CreateFile', 'RenameFile', 'DeleteFile', 'ChangeAnnotation', 'FileOperationFilter', 'FileRename', 'FileDelete', 'MonikerOptions', 'TypeHierarchyOptions', 'InlineValueContext', 'InlineValueText', 'InlineValueVariableLookup', 'InlineValueEvaluatableExpression', 'InlineValueOptions', 'InlayHintLabelPart', 'MarkupContent', 'InlayHintOptions', 'RelatedFullDocumentDiagnosticReport', 'RelatedUnchangedDocumentDiagnosticReport', 'FullDocumentDiagnosticReport', 'UnchangedDocumentDiagnosticReport', 'DiagnosticOptions', 'PreviousResultId', 'NotebookDocument', 'TextDocumentItem', 'VersionedNotebookDocumentIdentifier', 'NotebookDocumentChangeEvent', 'NotebookDocumentIdentifier', 'Registration', 'Unregistration', 'WorkspaceFoldersInitializeParams', 'ServerCapabilities', 'VersionedTextDocumentIdentifier', 'SaveOptions', 'FileEvent', 'FileSystemWatcher', 'Diagnostic', 'CompletionContext', 'CompletionItemLabelDetails', 'InsertReplaceEdit', 'CompletionOptions', 'HoverOptions', 'SignatureHelpContext', 'SignatureInformation', 'SignatureHelpOptions', 'DefinitionOptions', 'ReferenceContext', 'ReferenceOptions', 'DocumentHighlightOptions', 'BaseSymbolInformation', 'DocumentSymbolOptions', 'CodeActionContext', 'CodeActionOptions', 'WorkspaceSymbolOptions', 'CodeLensOptions', 'DocumentLinkOptions', 'FormattingOptions', 'DocumentFormattingOptions', 'DocumentRangeFormattingOptions', 'DocumentOnTypeFormattingOptions', 'RenameOptions', 'ExecuteCommandOptions', 'SemanticTokensLegend', 'OptionalVersionedTextDocumentIdentifier', 'AnnotatedTextEdit', 'ResourceOperation', 'CreateFileOptions', 'RenameFileOptions', 'DeleteFileOptions', 'FileOperationPattern', 'WorkspaceFullDocumentDiagnosticReport', 'WorkspaceUnchangedDocumentDiagnosticReport', 'NotebookCell', 'NotebookCellArrayChange', 'ClientCapabilities', 'TextDocumentSyncOptions', 'NotebookDocumentSyncOptions', 'NotebookDocumentSyncRegistrationOptions', 'WorkspaceFoldersServerCapabilities', 'FileOperationOptions', 'CodeDescription', 'DiagnosticRelatedInformation', 'ParameterInformation', 'NotebookCellTextDocumentFilter', 'FileOperationPatternOptions', 'ExecutionSummary', 'WorkspaceClientCapabilities', 'TextDocumentClientCapabilities', 'NotebookDocumentClientCapabilities', 'WindowClientCapabilities', 'GeneralClientCapabilities', 'RelativePattern', 'WorkspaceEditClientCapabilities', 'DidChangeConfigurationClientCapabilities', 'DidChangeWatchedFilesClientCapabilities', 'WorkspaceSymbolClientCapabilities', 'ExecuteCommandClientCapabilities', 'SemanticTokensWorkspaceClientCapabilities', 'CodeLensWorkspaceClientCapabilities', 'FileOperationClientCapabilities', 'InlineValueWorkspaceClientCapabilities', 'InlayHintWorkspaceClientCapabilities', 'DiagnosticWorkspaceClientCapabilities', 'TextDocumentSyncClientCapabilities', 'CompletionClientCapabilities', 'HoverClientCapabilities', 'SignatureHelpClientCapabilities', 'DeclarationClientCapabilities', 'DefinitionClientCapabilities', 'TypeDefinitionClientCapabilities', 'ImplementationClientCapabilities', 'ReferenceClientCapabilities', 'DocumentHighlightClientCapabilities', 'DocumentSymbolClientCapabilities', 'CodeActionClientCapabilities', 'CodeLensClientCapabilities', 'DocumentLinkClientCapabilities', 'DocumentColorClientCapabilities', 'DocumentFormattingClientCapabilities', 'DocumentRangeFormattingClientCapabilities', 'DocumentOnTypeFormattingClientCapabilities', 'RenameClientCapabilities', 'FoldingRangeClientCapabilities', 'SelectionRangeClientCapabilities', 'PublishDiagnosticsClientCapabilities', 'CallHierarchyClientCapabilities', 'SemanticTokensClientCapabilities', 'LinkedEditingRangeClientCapabilities', 'MonikerClientCapabilities', 'TypeHierarchyClientCapabilities', 'InlineValueClientCapabilities', 'InlayHintClientCapabilities', 'DiagnosticClientCapabilities', 'NotebookDocumentSyncClientCapabilities', 'ShowMessageRequestClientCapabilities', 'ShowDocumentClientCapabilities', 'RegularExpressionsClientCapabilities', 'MarkdownClientCapabilities', '__CodeActionClientCapabilities_codeActionLiteralSupport_Type_1', '__CodeActionClientCapabilities_codeActionLiteralSupport_codeActionKind_Type_1', '__CodeActionClientCapabilities_resolveSupport_Type_1', '__CodeAction_disabled_Type_1', '__CompletionClientCapabilities_completionItemKind_Type_1', '__CompletionClientCapabilities_completionItem_Type_1', '__CompletionClientCapabilities_completionItem_insertTextModeSupport_Type_1', '__CompletionClientCapabilities_completionItem_resolveSupport_Type_1', '__CompletionClientCapabilities_completionItem_tagSupport_Type_1', '__CompletionClientCapabilities_completionList_Type_1', '__CompletionList_itemDefaults_Type_1', '__CompletionList_itemDefaults_editRange_Type_1', '__CompletionOptions_completionItem_Type_1', '__CompletionOptions_completionItem_Type_2', '__DocumentSymbolClientCapabilities_symbolKind_Type_1', '__DocumentSymbolClientCapabilities_tagSupport_Type_1', '__FoldingRangeClientCapabilities_foldingRangeKind_Type_1', '__FoldingRangeClientCapabilities_foldingRange_Type_1', '__GeneralClientCapabilities_staleRequestSupport_Type_1', '__InitializeResult_serverInfo_Type_1', '__InlayHintClientCapabilities_resolveSupport_Type_1', '__MarkedString_Type_1', '__NotebookDocumentChangeEvent_cells_Type_1', '__NotebookDocumentChangeEvent_cells_structure_Type_1', '__NotebookDocumentChangeEvent_cells_textContent_Type_1', '__NotebookDocumentFilter_Type_1', '__NotebookDocumentFilter_Type_2', '__NotebookDocumentFilter_Type_3', '__NotebookDocumentSyncOptions_notebookSelector_Type_1', '__NotebookDocumentSyncOptions_notebookSelector_Type_2', '__NotebookDocumentSyncOptions_notebookSelector_Type_3', '__NotebookDocumentSyncOptions_notebookSelector_Type_4', '__NotebookDocumentSyncOptions_notebookSelector_cells_Type_1', '__NotebookDocumentSyncOptions_notebookSelector_cells_Type_2', '__NotebookDocumentSyncOptions_notebookSelector_cells_Type_3', '__NotebookDocumentSyncOptions_notebookSelector_cells_Type_4', '__PrepareRenameResult_Type_1', '__PrepareRenameResult_Type_2', '__PublishDiagnosticsClientCapabilities_tagSupport_Type_1', '__SemanticTokensClientCapabilities_requests_Type_1', '__SemanticTokensClientCapabilities_requests_full_Type_1', '__SemanticTokensOptions_full_Type_1', '__SemanticTokensOptions_full_Type_2', '__ServerCapabilities_workspace_Type_1', '__ShowMessageRequestClientCapabilities_messageActionItem_Type_1', '__SignatureHelpClientCapabilities_signatureInformation_Type_1', '__SignatureHelpClientCapabilities_signatureInformation_parameterInformation_Type_1', '__TextDocumentContentChangeEvent_Type_1', '__TextDocumentContentChangeEvent_Type_2', '__TextDocumentFilter_Type_1', '__TextDocumentFilter_Type_2', '__TextDocumentFilter_Type_3', '__WorkspaceEditClientCapabilities_changeAnnotationSupport_Type_1', '__WorkspaceSymbolClientCapabilities_resolveSupport_Type_1', '__WorkspaceSymbolClientCapabilities_symbolKind_Type_1', '__WorkspaceSymbolClientCapabilities_tagSupport_Type_1', '__WorkspaceSymbol_location_Type_1', '___InitializeParams_clientInfo_Type_1']
- variables: ['URI']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/solidlsp/lsp_protocol_handler/server.py

### Imports (resolved)
- root/serena/src/solidlsp/lsp_protocol_handler/lsp_types.py [import style: relative, symbol(s): ErrorCodes]

### Exports (symbols)
- classes: ['ProcessLaunchInfo', 'LSPError', 'StopLoopException', 'MessageType']
- functions: ['make_response', 'make_error_response', 'make_notification', 'make_request', 'create_message', 'content_length']
- variables: ['CONTENT_LENGTH', 'ENCODING']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes
- HTTP/API server implementation

---

## File: root/serena/src/solidlsp/settings.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['SolidLSPSettings']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/solidlsp/util/subprocess_util.py

### Imports (resolved)
- (none)

### Exports (symbols)
- functions: ['subprocess_kwargs']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/src/solidlsp/util/zip.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['SafeZipExtractor']

### Exports (files/artifacts written or produced)
- <dynamic path>; callsite: <function:99>; method: open

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/__init__.py

### Imports (resolved)
- (none)

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/conftest.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['LanguageParamRequest']
- functions: ['resources_dir', 'get_repo_path', 'create_ls', 'create_default_ls', 'create_default_project', 'repo_path', 'language_server', 'project']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/resources/repos/python/test_repo/custom_test/__init__.py

### Imports (resolved)
- (none)

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/resources/repos/python/test_repo/custom_test/advanced_features.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['UserDict', 'Status', 'Priority', 'Permissions', 'BaseProcessor', 'DataProcessor', 'Task', 'Repository', 'Serializable', 'OuterClass', 'Meta', 'WithMeta', 'TreeNode']
- functions: ['log_execution', 'transaction_context', 'advanced_search', 'create_processor', 'with_retry', 'unreliable_operation', 'process_validated_data', 'main']
- variables: ['T', 'K', 'V']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: False

### Notes

---

## File: root/serena/test/resources/repos/python/test_repo/examples/__init__.py

### Imports (resolved)
- (none)

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/resources/repos/python/test_repo/examples/user_management.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['UserStats', 'UserManager']
- functions: ['process_user_data', 'main']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: False

### Notes

---

## File: root/serena/test/resources/repos/python/test_repo/ignore_this_dir_with_postfix/ignored_module.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['UserStats', 'UserManager']
- functions: ['process_user_data', 'main']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: False

### Notes

---

## File: root/serena/test/resources/repos/python/test_repo/scripts/__init__.py

### Imports (resolved)
- (none)

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/resources/repos/python/test_repo/scripts/run_app.py

### Imports (resolved)
- (none)

### Exports (symbols)
- functions: ['parse_args', 'load_config', 'create_sample_users', 'create_sample_items', 'run_user_operations', 'run_item_operations', 'main']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: True

### Notes

---

## File: root/serena/test/resources/repos/python/test_repo/test_repo/__init__.py

### Imports (resolved)
- (none)

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/resources/repos/python/test_repo/test_repo/complex_types.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['CustomListInt', 'CustomTypedDict', 'Outer2', 'ComplexExtension']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/resources/repos/python/test_repo/test_repo/models.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['BaseModel', 'User', 'Item', 'Collection', 'Loggable', 'Serializable', 'Auditable', 'BaseService', 'DataService', 'NetworkService', 'DataSyncService', 'LoggableUser', 'TrackedItem']
- functions: ['create_user_object']
- variables: ['T']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/resources/repos/python/test_repo/test_repo/name_collisions.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['ClassWillBeOverwritten', 'ClassWillBeOverwritten']
- functions: ['func_using_overwritten_var', 'func_will_be_overwritten', 'func_will_be_overwritten', 'func_calling_overwritten_func', 'func_calling_overwritten_class']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/resources/repos/python/test_repo/test_repo/nested.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['OuterClass']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/resources/repos/python/test_repo/test_repo/nested_base.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['BaseModule', 'SubModule', 'FirstLevel', 'TwoLevel', 'ThreeLevel', 'GenericExtension']
- variables: ['T']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/resources/repos/python/test_repo/test_repo/overloaded.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['DataProcessor']
- functions: ['process_data', 'process_data', 'process_data', 'process_data']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/resources/repos/python/test_repo/test_repo/services.py

### Imports (resolved)
- root/serena/test/resources/repos/python/test_repo/test_repo/models.py [import style: relative, symbol(s): Item, User]

### Exports (symbols)
- classes: ['UserService', 'ItemService']
- functions: ['create_service_container']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/resources/repos/python/test_repo/test_repo/utils.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['ConfigManager', 'Timer']
- functions: ['setup_logging', 'log_execution', 'map_list', 'retry']
- variables: ['T', 'U']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/resources/repos/python/test_repo/test_repo/variables.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['VariableContainer', 'VariableDataclass']
- functions: ['use_module_variables']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/serena/__init__.py

### Imports (resolved)
- (none)

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/serena/config/__init__.py

### Imports (resolved)
- (none)

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/serena/config/test_serena_config.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestProjectConfigAutogenerate']

### Exports (files/artifacts written or produced)
- <python_file>.write_text(...); callsite: <function:46>; method: write_text
- <go_file>.write_text(...); callsite: <function:71>; method: write_text
- <gitignore>.write_text(...); callsite: <function:96>; method: write_text
- <ts_file>.write_text(...); callsite: <function:111>; method: write_text

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/serena/test_edit_marker.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestEditMarker']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/serena/test_mcp.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['MockAgent', 'BaseMockTool', 'BasicTool']
- functions: ['test_make_tool_basic', 'test_make_tool_execution', 'test_make_tool_no_params', 'test_make_tool_no_return_description', 'test_make_tool_parameter_not_in_docstring', 'test_make_tool_multiline_docstring', 'test_make_tool_capitalization_and_periods', 'test_make_tool_missing_apply', 'test_make_tool_descriptions', 'is_test_mock_class', 'test_make_tool_all_tools']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/serena/test_serena_agent.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestSerenaAgent']
- functions: ['serena_config', 'serena_agent']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/serena/test_symbol.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestSymbolNameMatching']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/serena/test_symbol_editing.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['LineChange', 'CodeDiff', 'EditingTest', 'DeleteSymbolTest', 'InsertInRelToSymbolTest', 'ReplaceBodyTest']
- functions: ['test_delete_symbol', 'test_insert_in_rel_to_symbol', 'test_insert_python_class_before', 'test_insert_python_class_after', 'test_replace_body']
- variables: ['PYTHON_TEST_REL_FILE_PATH', 'TYPESCRIPT_TEST_FILE', 'NEW_PYTHON_FUNCTION', 'NEW_PYTHON_CLASS_WITH_LEADING_NEWLINES', 'NEW_PYTHON_CLASS_WITH_TRAILING_NEWLINES', 'NEW_TYPESCRIPT_FUNCTION', 'NEW_PYTHON_VARIABLE', 'NEW_TYPESCRIPT_FUNCTION_AFTER', 'PYTHON_REPLACED_BODY', 'TYPESCRIPT_REPLACED_BODY']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/serena/test_text_utils.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestSearchText', 'TestSearchFiles', 'TestGlobMatch']
- functions: ['mock_reader_always_match']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/serena/test_tool_parameter_types.py

### Imports (resolved)
- (none)

### Exports (symbols)
- functions: ['test_all_tool_parameters_have_type']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/serena/util/test_exception.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestHeadlessEnvironmentDetection', 'TestShowFatalExceptionSafe']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/serena/util/test_file_system.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestGitignoreParser']

### Exports (files/artifacts written or produced)
- <root_gitignore>.write_text(...); callsite: <function:70>; method: write_text
- <src_gitignore>.write_text(...); callsite: <function:79>; method: write_text
- <src_lib_gitignore>.write_text(...); callsite: <function:89>; method: write_text
- <docs_gitignore>.write_text(...); callsite: <function:98>; method: write_text
- <gitignore>.write_text(...); callsite: <function:136>; method: write_text
- <gitignore>.write_text(...); callsite: <function:162>; method: write_text
- <gitignore>.write_text(...); callsite: <function:231>; method: write_text
- <gitignore>.write_text(...); callsite: <function:266>; method: write_text
- <gitignore>.write_text(...); callsite: <function:322>; method: write_text
- <gitignore>.write_text(...); callsite: <function:382>; method: write_text
- <gitignore>.write_text(...); callsite: <function:405>; method: write_text
- <gitignore>.write_text(...); callsite: <function:432>; method: write_text
- <gitignore>.write_text(...); callsite: <function:452>; method: write_text
- <gitignore>.write_text(...); callsite: <function:485>; method: write_text
- <gitignore>.write_text(...); callsite: <function:529>; method: write_text
- <gitignore>.write_text(...); callsite: <function:542>; method: write_text
- <gitignore>.write_text(...); callsite: <function:562>; method: write_text
- <gitignore>.write_text(...); callsite: <function:569>; method: write_text
- <gitignore>.write_text(...); callsite: <function:606>; method: write_text
- <gitignore>.write_text(...); callsite: <function:633>; method: write_text
- <gitignore>.write_text(...); callsite: <function:662>; method: write_text
- <gitignore>.write_text(...); callsite: <function:694>; method: write_text

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/bash/__init__.py

### Imports (resolved)
- (none)

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/bash/test_bash_basic.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestBashLanguageServerBasics']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/clojure/__init__.py

### Imports (resolved)
- (none)

### Exports (symbols)
- functions: ['_test_clojure_cli']
- variables: ['CLI_FAIL', 'TEST_APP_PATH', 'CORE_PATH', 'UTILS_PATH']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/clojure/test_clojure_basic.py

### Imports (resolved)
- root/serena/test/solidlsp/clojure/__init__.py [import style: relative, symbol(s): CLI_FAIL, CORE_PATH, UTILS_PATH]

### Exports (symbols)
- classes: ['TestLanguageServerBasics', 'TestProjectBasics']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/csharp/test_csharp_basic.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestCSharpLanguageServer', 'TestCSharpSolutionProjectOpening']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/dart/__init__.py

### Imports (resolved)
- (none)

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/dart/test_dart_basic.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestDartLanguageServer']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/elixir/__init__.py

### Imports (resolved)
- (none)

### Exports (symbols)
- functions: ['_test_nextls_available']
- variables: ['NEXTLS_UNAVAILABLE_REASON', 'NEXTLS_UNAVAILABLE']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/elixir/conftest.py

### Imports (resolved)
- (none)

### Exports (symbols)
- functions: ['ensure_elixir_test_repo_compiled', 'setup_elixir_test_environment', 'elixir_test_repo_path']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/elixir/test_elixir_basic.py

### Imports (resolved)
- root/serena/test/solidlsp/elixir/__init__.py [import style: relative, symbol(s): NEXTLS_UNAVAILABLE, NEXTLS_UNAVAILABLE_REASON]

### Exports (symbols)
- classes: ['TestElixirBasic']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/elixir/test_elixir_ignored_dirs.py

### Imports (resolved)
- root/serena/test/solidlsp/elixir/__init__.py [import style: relative, symbol(s): NEXTLS_UNAVAILABLE, NEXTLS_UNAVAILABLE_REASON]

### Exports (symbols)
- functions: ['ls_with_ignored_dirs', 'test_symbol_tree_ignores_dir', 'test_find_references_ignores_dir', 'test_refs_and_symbols_with_glob_patterns', 'test_default_ignored_directories', 'test_symbol_tree_excludes_build_dirs']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/elixir/test_elixir_integration.py

### Imports (resolved)
- root/serena/test/solidlsp/elixir/__init__.py [import style: relative, symbol(s): NEXTLS_UNAVAILABLE, NEXTLS_UNAVAILABLE_REASON]

### Exports (symbols)
- classes: ['TestElixirIntegration', 'TestElixirProject']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/elixir/test_elixir_symbol_retrieval.py

### Imports (resolved)
- root/serena/test/solidlsp/elixir/__init__.py [import style: relative, symbol(s): NEXTLS_UNAVAILABLE, NEXTLS_UNAVAILABLE_REASON]

### Exports (symbols)
- classes: ['TestElixirLanguageServerSymbols']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/erlang/__init__.py

### Imports (resolved)
- (none)

### Exports (symbols)
- functions: ['_test_erlang_ls_available']
- variables: ['ERLANG_LS_UNAVAILABLE_REASON', 'ERLANG_LS_UNAVAILABLE']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/erlang/conftest.py

### Imports (resolved)
- (none)

### Exports (symbols)
- functions: ['ensure_erlang_test_repo_compiled', 'setup_erlang_test_environment', 'erlang_test_repo_path']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/erlang/test_erlang_basic.py

### Imports (resolved)
- root/serena/test/solidlsp/erlang/__init__.py [import style: relative, symbol(s): ERLANG_LS_UNAVAILABLE, ERLANG_LS_UNAVAILABLE_REASON]

### Exports (symbols)
- classes: ['TestErlangLanguageServerBasics']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/erlang/test_erlang_ignored_dirs.py

### Imports (resolved)
- root/serena/test/solidlsp/erlang/__init__.py [import style: relative, symbol(s): ERLANG_LS_UNAVAILABLE, ERLANG_LS_UNAVAILABLE_REASON]

### Exports (symbols)
- functions: ['ls_with_ignored_dirs', 'test_symbol_tree_ignores_dir', 'test_find_references_ignores_dir', 'test_refs_and_symbols_with_glob_patterns', 'test_default_ignored_directories', 'test_symbol_tree_excludes_build_dirs', 'test_ignore_compiled_files', 'test_rebar_directories_ignored', 'test_document_symbols_ignores_dirs', 'test_erlang_specific_ignore_patterns']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/erlang/test_erlang_symbol_retrieval.py

### Imports (resolved)
- root/serena/test/solidlsp/erlang/__init__.py [import style: relative, symbol(s): ERLANG_LS_UNAVAILABLE, ERLANG_LS_UNAVAILABLE_REASON]

### Exports (symbols)
- classes: ['TestErlangLanguageServerSymbols']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/go/test_go_basic.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestGoLanguageServer']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/java/test_java_basic.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestJavaLanguageServer']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/kotlin/test_kotlin_basic.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestKotlinLanguageServer']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/lua/test_lua_basic.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestLuaLanguageServer']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/nix/test_nix_basic.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestNixLanguageServer']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/php/test_php_basic.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestPhpLanguageServer']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/python/test_python_basic.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestLanguageServerBasics', 'TestProjectBasics']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/python/test_retrieval_with_ignored_dirs.py

### Imports (resolved)
- (none)

### Exports (symbols)
- functions: ['ls_with_ignored_dirs', 'test_symbol_tree_ignores_dir', 'test_find_references_ignores_dir', 'test_refs_and_symbols_with_glob_patterns']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/python/test_symbol_retrieval.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestLanguageServerSymbols']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/ruby/test_ruby_basic.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestRubyLanguageServer']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/ruby/test_ruby_symbol_retrieval.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestRubyLanguageServerSymbols']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/rust/test_rust_2024_edition.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestRust2024EditionLanguageServer']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/rust/test_rust_basic.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestRustLanguageServer']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/swift/test_swift_basic.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestSwiftLanguageServerBasics', 'TestSwiftProjectBasics']
- variables: ['WINDOWS_SKIP', 'WINDOWS_SKIP_REASON']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/terraform/test_terraform_basic.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestLanguageServerBasics']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/typescript/test_typescript_basic.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestTypescriptLanguageServer']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/util/test_zip.py

### Imports (resolved)
- (none)

### Exports (symbols)
- functions: ['temp_zip_file', 'test_extract_all_success', 'test_include_patterns', 'test_exclude_patterns', 'test_include_and_exclude_patterns', 'test_skip_on_error', 'test_long_path_normalization']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/serena/test/solidlsp/zig/test_zig_basic.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestZigLanguageServer']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/src/hlynr_bridge/__init__.py

### Imports (resolved)
- root/src/hlynr_bridge/server.py [import style: relative, symbol(s): HlynrBridgeServer]
- root/src/hlynr_bridge/schemas.py [import style: relative, symbol(s): InferenceRequest, InferenceResponse, HealthResponse]
- root/src/hlynr_bridge/transforms.py [import style: relative, symbol(s): get_transform, validate_transform_version]
- root/src/hlynr_bridge/normalize.py [import style: relative, symbol(s): load_vecnorm, set_deterministic_inference_mode]
- root/src/hlynr_bridge/seed_manager.py [import style: relative, symbol(s): set_deterministic_seeds, get_current_seed]

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/src/hlynr_bridge/clamps.py

### Imports (resolved)
- root/src/hlynr_bridge/schemas.py [import style: relative, symbol(s): RateCommand, ActionCommand, SafetyInfo]

### Exports (symbols)
- classes: ['SafetyLimits', 'SafetyClampSystem']
- functions: ['get_safety_clamp_system']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/src/hlynr_bridge/config.py

### Imports (resolved)
- root/src/phase4_rl/config.py [import style: absolute, symbol(s): *]

### Exports (symbols)
- functions: ['get_config', 'reset_config']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/src/hlynr_bridge/episode_logger.py

### Imports (resolved)
- root/src/phase4_rl/episode_logger.py [import style: absolute, symbol(s): *]

### Exports (symbols)
- functions: ['get_inference_logger', 'reset_inference_logger']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes
- Episode logging and JSONL generation

---

## File: root/src/hlynr_bridge/normalize.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['VecNormalizeInfo', 'VecNormalizeManager']
- functions: ['get_vecnorm_manager', 'load_vecnormalize_by_id', 'load_vecnorm', 'set_deterministic_inference_mode', 'register_vecnormalize_from_checkpoint']

### Exports (files/artifacts written or produced)
- <vecnorm>.save(...); callsite: <function:141>; method: save
- <dynamic path>; callsite: <function:78>; method: open
- <json>.dump(...); callsite: <function:79>; method: dump

### Entry Points
- main guard: False
- argparse: False

### Notes
- VecNormalize statistics handling

---

## File: root/src/hlynr_bridge/schemas.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['FrameInfo', 'MetaInfo', 'BlueState', 'RedState', 'GuidanceInfo', 'EnvironmentInfo', 'NormalizationInfo', 'InferenceRequest', 'RateCommand', 'ActionCommand', 'ClipFractions', 'DiagnosticsInfo', 'SafetyInfo', 'InferenceResponse', 'HealthResponse', 'MetricsResponse']
- functions: ['validate_obs_version', 'validate_transform_version', 'validate_rate_commands', 'check_if_clamped']
- variables: ['SUPPORTED_OBS_VERSIONS', 'SUPPORTED_TRANSFORM_VERSIONS', 'MAX_RATE_RADPS', 'MIN_RATE_RADPS', 'CANONICAL_CLAMP_REASONS']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: False

### Notes
- API schema definitions

---

## File: root/src/hlynr_bridge/seed_manager.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['SeedManager']
- functions: ['get_seed_manager', 'set_deterministic_seeds', 'get_current_seed', 'validate_deterministic_setup', 'enforce_single_thread_inference', 'validate_seed_env_var']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: False

### Notes

---

## File: root/src/hlynr_bridge/server.py

### Imports (resolved)
- root/src/phase4_rl/env_config.py [import style: absolute, symbol(s): BridgeServerConfig]
- root/src/hlynr_bridge/schemas.py [import style: relative, symbol(s): InferenceRequest, InferenceResponse, HealthResponse, MetricsResponse, ActionCommand, RateCommand, DiagnosticsInfo, SafetyInfo, ClipFractions]
- root/src/hlynr_bridge/transforms.py [import style: absolute, symbol(s): get_transform, validate_transform_version]
- root/src/hlynr_bridge/normalize.py [import style: relative, symbol(s): get_vecnorm_manager, load_vecnormalize_by_id]
- root/src/hlynr_bridge/seed_manager.py [import style: relative, symbol(s): set_deterministic_seeds, get_current_seed, validate_deterministic_setup]
- root/src/hlynr_bridge/clamps.py [import style: relative, symbol(s): get_safety_clamp_system, SafetyLimits]
- root/src/hlynr_bridge/episode_logger.py [import style: relative, symbol(s): get_inference_logger, reset_inference_logger]
- root/src/hlynr_bridge/config.py [import style: relative, symbol(s): get_config, reset_config]
- root/src/phase4_rl/scenarios/__init__.py [import style: absolute, symbol(s): get_scenario_loader, reset_scenario_loader]
- root/src/phase4_rl/radar_env.py [import style: absolute, symbol(s): RadarEnv]
- root/src/phase4_rl/fast_sim_env.py [import style: absolute, symbol(s): FastSimEnv]
- root/src/phase4_rl/env_config.py [import style: absolute, symbol(s): get_bridge_config, BridgeServerConfig]
- root/src/hlynr_bridge/clamps.py [import style: relative, symbol(s): get_safety_clamp_system, SafetyLimits]
- root/src/hlynr_bridge/episode_logger.py [import style: relative, symbol(s): get_inference_logger, reset_inference_logger]
- root/src/hlynr_bridge/config.py [import style: relative, symbol(s): get_config, reset_config]
- root/src/phase4_rl/scenarios/__init__.py [import style: absolute, symbol(s): get_scenario_loader, reset_scenario_loader]
- root/src/phase4_rl/radar_env.py [import style: absolute, symbol(s): RadarEnv]
- root/src/phase4_rl/fast_sim_env.py [import style: absolute, symbol(s): FastSimEnv]
- root/src/phase4_rl/env_config.py [import style: absolute, symbol(s): get_bridge_config, BridgeServerConfig]

### Exports (symbols)
- classes: ['HlynrBridgeServer']
- functions: ['ensure_model_loaded', 'v1_inference', 'health_check', 'get_metrics', '_get_json_metrics', 'legacy_act', '_parse_canonical_clamp_reasons', '_error_response', '_validate_request_versions', '_unity_to_rl_observation', '_rl_to_unity_action', '_compute_diagnostics', '_update_server_stats', '_log_inference_step', 'verify_manifest', 'main']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: True

### Notes
- HTTP/API server implementation

---

## File: root/src/hlynr_bridge/transforms.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TransformVersion', 'CoordinateTransform']
- functions: ['get_transform', 'validate_transform_version']
- variables: ['TRANSFORM_VERSIONS']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: False

### Notes
- Contains coordinate transforms (ENU↔Unity)

---

## File: root/src/phase4_rl/__init__.py

### Imports (resolved)
- root/src/phase4_rl/radar_env.py [import style: relative, symbol(s): RadarEnv]
- root/src/phase4_rl/config.py [import style: relative, symbol(s): ConfigLoader, get_config]
- root/src/phase4_rl/scenarios/__init__.py [import style: relative, symbol(s): ScenarioLoader, get_scenario_loader]
- root/src/phase4_rl/diagnostics.py [import style: relative, symbol(s): Logger, export_to_csv, export_to_json, plot_metrics]
- root/src/phase4_rl/train_radar_ppo.py [import style: relative, symbol(s): Phase4Trainer]
- root/src/phase4_rl/run_inference.py [import style: relative, symbol(s): Phase4InferenceRunner]

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/src/phase4_rl/bridge_server.py

### Imports (resolved)
- root/src/phase4_rl/config.py [import style: relative, symbol(s): get_config, reset_config]
- root/src/phase4_rl/scenarios/__init__.py [import style: relative, symbol(s): get_scenario_loader, reset_scenario_loader]
- root/src/phase4_rl/radar_env.py [import style: relative, symbol(s): RadarEnv]
- root/src/phase4_rl/fast_sim_env.py [import style: relative, symbol(s): FastSimEnv]

### Exports (symbols)
- classes: ['BridgeServer']
- functions: ['health_check', 'get_stats', 'get_action', 'reset_environment', 'not_found', 'internal_error', 'main']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: True

### Notes
- HTTP/API server implementation

---

## File: root/src/phase4_rl/clamps.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['SafetyLimits', 'ClampResult', 'SafetyClampSystem']
- functions: ['get_safety_clamp_system', 'apply_safety_clamps']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: False

### Notes

---

## File: root/src/phase4_rl/client_stub.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['BridgeClient']
- functions: ['generate_dummy_observation', 'test_basic_functionality', 'test_inference', 'test_error_handling', 'benchmark_performance', 'main']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: True

### Notes

---

## File: root/src/phase4_rl/config.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['ConfigLoader']
- functions: ['get_config', 'reset_config']

### Exports (files/artifacts written or produced)
- <dynamic path>; callsite: <function:184>; method: open
- <yaml>.dump(...); callsite: <function:185>; method: dump

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/src/phase4_rl/debug_config.py

### Imports (resolved)
- (none)

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/src/phase4_rl/demo_episode_logging.py

### Imports (resolved)
- (none)

### Exports (symbols)
- functions: ['demo_training_with_logging', 'demo_inference_with_logging', 'examine_log_structure', 'unity_integration_notes', 'main']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: True

### Notes

---

## File: root/src/phase4_rl/diagnostics.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['Logger']
- functions: ['export_to_csv', 'export_to_json', 'load_episode_data', 'load_inference_results', '_migrate_episode_data_v0_to_v1', '_migrate_inference_results_v0_to_v1', 'plot_metrics', '_plot_success_rates', '_plot_reward_distributions', '_plot_episode_lengths', '_plot_trajectories', '_plot_performance_correlation', 'export']

### Exports (files/artifacts written or produced)
- <dynamic path>; callsite: <function:374>; method: open
- <json>.dump(...); callsite: <function:375>; method: dump
- <dynamic path>; callsite: <function:291>; method: open
- <json>.dump(...); callsite: <function:292>; method: dump
- <dynamic path>; callsite: <function:346>; method: open

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/src/phase4_rl/env_config.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['BridgeServerConfig']
- functions: ['load_dotenv_if_available', 'get_bridge_config', 'get_cached_bridge_config', 'reset_bridge_config_cache']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: False

### Notes

---

## File: root/src/phase4_rl/episode_logger.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['AgentState', 'Event', 'EpisodeLogger']

### Exports (files/artifacts written or produced)
- <dynamic path>; callsite: <function:326>; method: open
- <json>.dump(...); callsite: <function:327>; method: dump
- <dynamic path>; callsite: <function:333>; method: open
- <json>.dump(...); callsite: <function:334>; method: dump

### Entry Points
- main guard: False
- argparse: False

### Notes
- Episode logging and JSONL generation

---

## File: root/src/phase4_rl/fast_sim_env.py

### Imports (resolved)
- root/src/phase4_rl/radar_env.py [import style: relative, symbol(s): RadarEnv]
- root/src/phase4_rl/config.py [import style: relative, symbol(s): get_config]
- root/src/phase4_rl/scenarios/__init__.py [import style: relative, symbol(s): get_scenario_loader]
- root/src/phase4_rl/episode_logger.py [import style: relative, symbol(s): EpisodeLogger]

### Exports (symbols)
- classes: ['FastSimEnv']
- functions: ['make_fast_sim_env']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: False

### Notes

---

## File: root/src/phase4_rl/inference_logger.py

### Imports (resolved)
- root/src/phase4_rl/episode_logger.py [import style: relative, symbol(s): EpisodeLogger, Event]

### Exports (symbols)
- classes: ['InferenceLogEntry', 'InferenceEpisodeLogger']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/src/phase4_rl/plot_episode.py

### Imports (resolved)
- (none)

### Exports (symbols)
- functions: ['load_episode_data', 'plot_trajectories', 'plot_reward_over_time', 'plot_distances', 'plot_summary_dashboard', 'main']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: True

### Notes

---

## File: root/src/phase4_rl/radar_env.py

### Imports (resolved)
- root/src/phase4_rl/config.py [import style: relative, symbol(s): get_config]
- root/src/phase4_rl/scenarios/__init__.py [import style: relative, symbol(s): get_scenario_loader]

### Exports (symbols)
- classes: ['RadarEnv']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/src/phase4_rl/run_inference.py

### Imports (resolved)
- root/src/phase4_rl/config.py [import style: relative, symbol(s): get_config, reset_config]
- root/src/phase4_rl/scenarios/__init__.py [import style: relative, symbol(s): get_scenario_loader, reset_scenario_loader]
- root/src/phase4_rl/radar_env.py [import style: relative, symbol(s): RadarEnv]
- root/src/phase4_rl/diagnostics.py [import style: relative, symbol(s): Logger, export_to_csv, export_to_json, plot_metrics]

### Exports (symbols)
- classes: ['Phase4InferenceRunner']
- functions: ['main']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: True

### Notes

---

## File: root/src/phase4_rl/scenarios/__init__.py

### Imports (resolved)
- root/src/phase4_rl/scenarios/scenarios.py [import style: relative, symbol(s): ScenarioLoader, get_scenario_loader, reset_scenario_loader]

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/src/phase4_rl/scenarios/scenarios.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['ScenarioLoader']
- functions: ['get_scenario_loader', 'reset_scenario_loader']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/src/phase4_rl/stress_bridge.py

### Imports (resolved)
- root/src/phase4_rl/client_stub.py [import style: relative, symbol(s): BridgeClient, generate_dummy_observation]

### Exports (symbols)
- classes: ['StressTestConfig', 'RequestResult', 'StressTestRunner']
- functions: ['save_results', 'main']

### Exports (files/artifacts written or produced)
- <dynamic path>; callsite: <function:460>; method: open
- <json>.dump(...); callsite: <function:461>; method: dump

### Entry Points
- main guard: True
- argparse: True

### Notes

---

## File: root/src/phase4_rl/test_6dof_simple.py

### Imports (resolved)
- (none)

### Exports (symbols)
- functions: ['test_6dof']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: False

### Notes

---

## File: root/src/phase4_rl/test_bridge_basic.py

### Imports (resolved)
- (none)

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/src/phase4_rl/test_episode_logging.py

### Imports (resolved)
- (none)

### Exports (symbols)
- functions: ['test_episode_logging']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: False

### Notes

---

## File: root/src/phase4_rl/test_integration.py

### Imports (resolved)
- (none)

### Exports (symbols)
- functions: ['test_imports', 'test_configuration', 'test_scenarios', 'test_environment', 'test_diagnostics', 'test_integration', 'main']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: False

### Notes

---

## File: root/src/phase4_rl/test_random_baseline.py

### Imports (resolved)
- (none)

### Exports (symbols)
- functions: ['test_random_baseline_single_seed', 'test_random_baseline_multi_seed', 'calculate_performance_improvement', 'validate_trained_model', 'save_baseline_results', 'load_baseline_results', 'main', 'test_random_baseline']

### Exports (files/artifacts written or produced)
- <dynamic path>; callsite: <function:283>; method: open
- <json>.dump(...); callsite: <function:284>; method: dump

### Entry Points
- main guard: True
- argparse: True

### Notes

---

## File: root/src/phase4_rl/tests/__init__.py

### Imports (resolved)
- (none)

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/src/phase4_rl/tests/test_best_ckpt_exists.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestBestModelCallback', 'TestBestModelIntegration']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: False

### Notes

---

## File: root/src/phase4_rl/tests/test_callbacks_functional.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestCallbackInstantiation', 'TestEntropyScheduleLogic', 'TestCallbacksConfigCompatibility', 'TestBestModelMetadata', 'TestCallbackIntegration']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: False

### Notes

---

## File: root/src/phase4_rl/tests/test_config.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestConfigLoader']

### Exports (files/artifacts written or produced)
- <yaml>.dump(...); callsite: <function:50>; method: dump
- <config>.save(...); callsite: <function:113>; method: save

### Entry Points
- main guard: True
- argparse: False

### Notes

---

## File: root/src/phase4_rl/tests/test_diagnostics_json.py

### Imports (resolved)
- (none)

### Exports (symbols)
- functions: ['test_json_record_reload_cycle', 'test_empty_logger_handling', 'test_metrics_calculation']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: False

### Notes

---

## File: root/src/phase4_rl/tests/test_diagnostics_schema.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestDiagnosticsSchema']

### Exports (files/artifacts written or produced)
- <json>.dump(...); callsite: <function:129>; method: dump
- <json>.dump(...); callsite: <function:154>; method: dump
- <json>.dump(...); callsite: <function:188>; method: dump
- <json>.dump(...); callsite: <function:215>; method: dump
- <json>.dump(...); callsite: <function:275>; method: dump

### Entry Points
- main guard: True
- argparse: False

### Notes
- API schema definitions

---

## File: root/src/phase4_rl/tests/test_end_to_end_api.py

### Imports (resolved)
- (none)

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/src/phase4_rl/tests/test_entropy_integration.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestEntropyIntegration']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: False

### Notes

---

## File: root/src/phase4_rl/tests/test_entropy_schedule.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestEntropyScheduleCallback', 'TestEntropyScheduleIntegration']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: False

### Notes

---

## File: root/src/phase4_rl/tests/test_interface_compat.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestRadarEnvInterface', 'TestEnvironmentComparison', 'TestEnvironmentStability', 'TestEnvironmentSeeding']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: False

### Notes

---

## File: root/src/phase4_rl/tests/test_multi_entity.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestMultiEntityEnvironment', 'TestEntityInteraction', 'TestScenarioEntityConfiguration', 'TestEntityPhysics']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: False

### Notes

---

## File: root/src/phase4_rl/tests/test_normalization_determinism.py

### Imports (resolved)
- (none)

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/src/phase4_rl/tests/test_radar_obs.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestRadarOnlyObservations']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: False

### Notes

---

## File: root/src/phase4_rl/tests/test_safety_clamps.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestSafetyClamps']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/src/phase4_rl/tests/test_scenarios.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestScenarioLoader', 'TestScenarioValidation']

### Exports (files/artifacts written or produced)
- <dynamic path>; callsite: <function:65>; method: open
- <json>.dump(...); callsite: <function:66>; method: dump
- <dynamic path>; callsite: <function:75>; method: open
- <json>.dump(...); callsite: <function:76>; method: dump
- <dynamic path>; callsite: <function:95>; method: open
- <json>.dump(...); callsite: <function:96>; method: dump

### Entry Points
- main guard: True
- argparse: False

### Notes

---

## File: root/src/phase4_rl/tests/test_schemas_validation.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestSchemaValidation']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes
- API schema definitions

---

## File: root/src/phase4_rl/tests/test_seed_fixed.py

### Imports (resolved)
- (none)

### Exports (symbols)
- functions: ['test_seeding_fix', 'test_step_reproducibility_fixed']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: False

### Notes

---

## File: root/src/phase4_rl/tests/test_stress_bridge_stub.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['TestBridgeStressStub']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: False

### Notes

---

## File: root/src/phase4_rl/tests/test_transforms_comprehensive.py

### Imports (resolved)
- (none)

### Exports (symbols)
- (none)

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: False
- argparse: False

### Notes
- Contains coordinate transforms (ENU↔Unity)

---

## File: root/src/phase4_rl/train_radar_ppo.py

### Imports (resolved)
- root/src/phase4_rl/config.py [import style: relative, symbol(s): get_config, reset_config]
- root/src/phase4_rl/scenarios/__init__.py [import style: relative, symbol(s): get_scenario_loader, reset_scenario_loader]
- root/src/phase4_rl/radar_env.py [import style: relative, symbol(s): RadarEnv]
- root/src/phase4_rl/fast_sim_env.py [import style: relative, symbol(s): FastSimEnv]
- root/src/phase4_rl/training_callbacks.py [import style: relative, symbol(s): EntropyScheduleCallback, LearningRateSchedulerCallback, BestModelCallback, ClipRangeAdaptiveCallback]

### Exports (symbols)
- classes: ['Phase4Trainer']
- functions: ['main']

### Exports (files/artifacts written or produced)
- <yaml>.dump(...); callsite: <function:509>; method: dump
- <dynamic path>; callsite: <function:375>; method: open
- <json>.dump(...); callsite: <function:376>; method: dump

### Entry Points
- main guard: True
- argparse: True

### Notes

---

## File: root/src/phase4_rl/training_callbacks.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['EntropyScheduleCallback', 'LearningRateSchedulerCallback', 'BestModelCallback', 'ClipRangeAdaptiveCallback']

### Exports (files/artifacts written or produced)
- <dynamic path>; callsite: <function:346>; method: open
- <json>.dump(...); callsite: <function:347>; method: dump

### Entry Points
- main guard: False
- argparse: False

### Notes

---

## File: root/src/phase4_rl/validate_performance.py

### Imports (resolved)
- (none)

### Exports (symbols)
- classes: ['PerformanceMetrics', 'PerformanceValidator']
- functions: ['main']

### Exports (files/artifacts written or produced)
- <dynamic path>; callsite: <function:408>; method: open
- <json>.dump(...); callsite: <function:409>; method: dump

### Entry Points
- main guard: True
- argparse: True

### Notes

---

## File: root/test_vecnorm_blocker.py

### Imports (resolved)
- (none)

### Exports (symbols)
- functions: ['create_mock_env', 'test_vecnorm_deterministic_loading']

### Exports (files/artifacts written or produced)
- (none)

### Entry Points
- main guard: True
- argparse: False

### Notes
- VecNormalize statistics handling

---

