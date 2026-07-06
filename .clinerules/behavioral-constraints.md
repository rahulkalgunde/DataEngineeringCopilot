# Behavioral Constraints for Low-Power Execution

You are operating under limited compute resources. You MUST adhere strictly to these rules to avoid tool crashes and logic drift:

## Core Operational Protocols
1. **One Edit Per Turn:** Never attempt to modify more than ONE file at a time. Do not chain multiple write actions together.
2. **Read Before Writing:** You must use `read_file` to review a file completely before applying a edit. Never guess the contents of a file.
3. **No Speculative Code:** Do not write boilerplate or placeholder comments like `# TODO: implement later`. Write the complete, production-ready implementation immediately.
4. **No Unasked Refactoring:** Fix only the explicit target requested. Do not clean up, rename, or touch surrounding functions unless explicitly instructed.
5. **Acknowledge and Test:** After making an edit, stop and ask the user to verify or run tests. Do not proceed to subsequent steps automatically.
6. **Test-Driven Implementation** When you write new code or drastically alter an existing module, you need to write the unit tests first. After you do code change, test code using unit test.
7. **Break complex lengthy tasks into smaller iterations** Do not perform single long big lengthy task. Break it into smaller managables tasks. Report task progress after each task.