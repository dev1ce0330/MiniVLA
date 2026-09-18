import torch


class ActionTokenizer:

    def __init__(
        self,
        tokenizer,
        num_bins=256,
        action_q01=None,
        action_q99=None,
    ):

        self.tokenizer = tokenizer
        self.num_bins = num_bins

        # ====================================================
        # 1. Action tokens
        # ====================================================

        self.action_tokens = [
            f"<ACT_{i:03d}>"
            for i in range(num_bins)
        ]

        self.gripper_open_token = (
            "<GRIP_OPEN>"
        )

        self.gripper_close_token = (
            "<GRIP_CLOSE>"
        )

        self.special_tokens = (
            self.action_tokens
            + [
                self.gripper_open_token,
                self.gripper_close_token,
            ]
        )

        tokenizer.add_special_tokens(
            {
                "additional_special_tokens":
                    self.special_tokens
            }
        )

        # ====================================================
        # 2. Token IDs
        # ====================================================

        self.action_token_ids = (
            tokenizer
            .convert_tokens_to_ids(
                self.action_tokens
            )
        )

        self.gripper_open_id = (
            tokenizer
            .convert_tokens_to_ids(
                self.gripper_open_token
            )
        )

        self.gripper_close_id = (
            tokenizer
            .convert_tokens_to_ids(
                self.gripper_close_token
            )
        )

        # token id -> bin index
        self.id_to_bin = {
            token_id: bin_index
            for bin_index, token_id
            in enumerate(
                self.action_token_ids
            )
        }

        # ====================================================
        # 3. Per-dimension normalization stats
        # ====================================================

        if action_q01 is None:

            action_q01 = [
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
            ]

        if action_q99 is None:

            action_q99 = [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ]

        assert len(action_q01) == 6
        assert len(action_q99) == 6

        self.action_q01 = torch.tensor(
            action_q01,
            dtype=torch.float32,
        )

        self.action_q99 = torch.tensor(
            action_q99,
            dtype=torch.float32,
        )

        assert torch.all(
            self.action_q99
            > self.action_q01
        )

    # ========================================================
    # Normalize raw action -> [-1, 1]
    # ========================================================

    def normalize_action(
        self,
        action,
    ):

        q01 = self.action_q01.to(
            device=action.device,
            dtype=action.dtype,
        )

        q99 = self.action_q99.to(
            device=action.device,
            dtype=action.dtype,
        )

        # -----------------------------------------------
        # clip到训练集q01/q99
        # -----------------------------------------------

        action = torch.maximum(
            action,
            q01,
        )

        action = torch.minimum(
            action,
            q99,
        )

        # -----------------------------------------------
        # [q01, q99]
        #
        #      ↓
        #
        # [-1, 1]
        # -----------------------------------------------

        normalized = (
            2.0
            * (
                action
                - q01
            )
            / (
                q99
                - q01
            )
            - 1.0
        )

        return normalized

    # ========================================================
    # [-1,1] -> raw LIBERO action
    # ========================================================

    def denormalize_action(
        self,
        normalized_action,
    ):

        q01 = self.action_q01.to(
            device=
                normalized_action.device,

            dtype=
                normalized_action.dtype,
        )

        q99 = self.action_q99.to(
            device=
                normalized_action.device,

            dtype=
                normalized_action.dtype,
        )

        raw_action = (
            q01
            + (
                normalized_action
                + 1.0
            )
            / 2.0
            * (
                q99
                - q01
            )
        )

        return raw_action

    # ========================================================
    # Encode
    # ========================================================

    def encode(
        self,
        action,
    ):

        # 支持：
        #
        # [7]
        # 或
        # [B, 7]

        if action.ndim == 1:

            action = action.unsqueeze(
                0
            )

        assert action.shape[-1] == 7

        # ----------------------------------------------------
        # First six continuous dimensions
        # ----------------------------------------------------

        continuous_action = (
            action[
                :,
                :6
            ]
        )

        gripper_action = (
            action[
                :,
                6
            ]
        )

        # raw action
        #
        # ↓
        #
        # normalized [-1,1]

        normalized_action = (
            self.normalize_action(
                continuous_action
            )
        )

        # ----------------------------------------------------
        # [-1,1]
        #
        # ↓
        #
        # bin 0~255
        # ----------------------------------------------------

        bin_indices = torch.round(
            (
                normalized_action
                + 1.0
            )
            / 2.0
            * (
                self.num_bins
                - 1
            )
        ).long()

        bin_indices = bin_indices.clamp(
            0,
            self.num_bins - 1,
        )

        # ----------------------------------------------------
        # bin index -> vocab token ID
        # ----------------------------------------------------

        action_token_id_tensor = (
            torch.tensor(
                self.action_token_ids,
                dtype=torch.long,
                device=action.device,
            )
        )

        continuous_token_ids = (
            action_token_id_tensor[
                bin_indices
            ]
        )

        # ----------------------------------------------------
        # Gripper
        #
        # 保持原来的规则：
        #
        # <= 0 -> OPEN
        # > 0  -> CLOSE
        # ----------------------------------------------------

        gripper_token_ids = (
            torch.where(
                gripper_action > 0,
                torch.tensor(
                    self.gripper_close_id,
                    device=action.device,
                    dtype=torch.long,
                ),
                torch.tensor(
                    self.gripper_open_id,
                    device=action.device,
                    dtype=torch.long,
                ),
            )
        )

        gripper_token_ids = (
            gripper_token_ids
            .unsqueeze(-1)
        )

        # ----------------------------------------------------
        # [B,6] + [B,1]
        #
        # ↓
        #
        # [B,7]
        # ----------------------------------------------------

        token_ids = torch.cat(
            [
                continuous_token_ids,
                gripper_token_ids,
            ],
            dim=-1,
        )

        return token_ids

    # ========================================================
    # Decode
    # ========================================================

    def decode(
        self,
        token_ids,
    ):

        if token_ids.ndim == 1:

            token_ids = (
                token_ids.unsqueeze(0)
            )

        assert (
            token_ids.shape[-1]
            == 7
        )

        device = token_ids.device

        batch_size = (
            token_ids.shape[0]
        )

        # ----------------------------------------------------
        # vocab ID -> bin index
        # ----------------------------------------------------

        bin_indices = torch.zeros(
            (
                batch_size,
                6,
            ),
            dtype=torch.long,
            device=device,
        )

        for batch_index in range(
            batch_size
        ):

            for dim in range(6):

                token_id = int(
                    token_ids[
                        batch_index,
                        dim
                    ].item()
                )

                bin_indices[
                    batch_index,
                    dim
                ] = self.id_to_bin[
                    token_id
                ]

        # ----------------------------------------------------
        # bin -> normalized [-1,1]
        # ----------------------------------------------------

        normalized_action = (
            2.0
            * bin_indices.float()
            / (
                self.num_bins
                - 1
            )
            - 1.0
        )

        # ----------------------------------------------------
        # normalized -> raw LIBERO action
        # ----------------------------------------------------

        continuous_action = (
            self.denormalize_action(
                normalized_action
            )
        )

        # ----------------------------------------------------
        # Gripper
        # ----------------------------------------------------

        gripper_ids = (
            token_ids[
                :,
                6
            ]
        )

        gripper_action = torch.where(
            gripper_ids
            == self.gripper_close_id,

            torch.tensor(
                1.0,
                device=device,
            ),

            torch.tensor(
                -1.0,
                device=device,
            ),
        )

        gripper_action = (
            gripper_action
            .unsqueeze(-1)
        )

        # ----------------------------------------------------
        # [B,6] + [B,1]
        #
        # ↓
        #
        # raw [B,7]
        # ----------------------------------------------------

        action = torch.cat(
            [
                continuous_action,
                gripper_action,
            ],
            dim=-1,
        )

        return action